using Baballonia.Services.events;
using Baballonia.Services.Inference.Enums;
using Microsoft.Extensions.Logging;
using OpenCvSharp;
using System;
using System.Diagnostics;

namespace Baballonia.Services.Inference;

public class EyeProcessingPipeline(IEyePipelineEventBus eyePipelineEventBus, ILogger<EyeProcessingPipeline> logger) : DefaultProcessingPipeline, IDisposable
{
    private readonly ImageCollector _imageCollector = new();

    // Per-eye corruption filter settings
    public bool CorruptionFilterEnabled { get; set; } = true;
    public double CorruptionThreshold { get; set; } = 0.022669;
    public bool CorruptionAdaptive { get; set; } = true;

    public bool StabilizeEyes { get; set; } = true;

    // Per-eye last-good frame cache
    private Mat? _lastGoodLeft;
    private Mat? _lastGoodRight;

    // Detector is recreated when threshold/adaptive changes
    private FastCorruptionDetector.FastCorruptionDetector? _perEyeDetector;

    // Stats
    private int _totalFrames;
    private int _repairedLeft;
    private int _repairedRight;

    // Rate-limited logging state
    private readonly Stopwatch _logTimer = Stopwatch.StartNew();
    private const double SummaryIntervalSeconds = 30.0;

    public void ResetDetector()
    {
        _perEyeDetector = new FastCorruptionDetector.FastCorruptionDetector(CorruptionThreshold, CorruptionAdaptive);
    }

    private FastCorruptionDetector.FastCorruptionDetector GetDetector()
    {
        return _perEyeDetector ??= new FastCorruptionDetector.FastCorruptionDetector(CorruptionThreshold, CorruptionAdaptive);
    }

    public float[]? RunUpdate()
    {
        var frame = VideoSource?.GetFrame(ColorType.Gray8);
        if(frame == null)
            return null;

        eyePipelineEventBus.Publish(new EyePipelineEvents.NewFrameEvent(frame));

        var transformed = ImageTransformer?.Apply(frame);
        frame.Dispose();
        if(transformed == null)
            return null;

        if (CorruptionFilterEnabled)
        {
            transformed = ApplyPerEyeCorruptionFilter(transformed);
            if (transformed == null)
                return null;
        }

        eyePipelineEventBus.Publish(new EyePipelineEvents.NewTransformedFrameEvent(transformed));

        var collected = _imageCollector.Apply(transformed);
        transformed.Dispose();
        if (collected == null)
            return null;

        if (InferenceService == null)
            return null;

        ImageConverter?.Convert(collected, InferenceService.GetInputTensor());

        var inferenceResult = InferenceService?.Run();
        if(inferenceResult == null)
            return null;

        if (Filter != null)
        {
            inferenceResult = Filter.Filter(inferenceResult);
        }

        ProcessExpressions(ref inferenceResult);

        eyePipelineEventBus.Publish(new EyePipelineEvents.NewFilteredResultEvent(inferenceResult));

        return inferenceResult;
    }

    /// <summary>
    /// Splits the 2-channel transformed Mat into left/right eye Mats, performs per-eye corruption
    /// detection, substitutes the last-known-good image for any corrupted eye, and re-merges.
    /// The last-good cache is only updated when the frame is accepted (not corrupted).
    /// </summary>
    private Mat? ApplyPerEyeCorruptionFilter(Mat transformed)
    {
        _totalFrames++;

        Mat[] channels = transformed.Split();
        if (channels.Length < 2)
        {
            transformed.Dispose();
            foreach (var ch in channels) ch.Dispose();
            return null;
        }

        var left = channels[0];
        var right = channels[1];

        var result = GetDetector().ProcessFramePair(left, right);

        bool leftRepaired = false;
        bool rightRepaired = false;

        if (result.LeftCorrupted)
        {
            if (_lastGoodLeft != null)
            {
                left.Dispose();
                left = _lastGoodLeft.Clone();
                leftRepaired = true;
                _repairedLeft++;
            }
            // else: no previous good frame available; pass the corrupted frame through
        }
        else
        {
            // Accept this frame as the new last-good reference
            _lastGoodLeft?.Dispose();
            _lastGoodLeft = left.Clone();
        }

        if (result.RightCorrupted)
        {
            if (_lastGoodRight != null)
            {
                right.Dispose();
                right = _lastGoodRight.Clone();
                rightRepaired = true;
                _repairedRight++;
            }
        }
        else
        {
            _lastGoodRight?.Dispose();
            _lastGoodRight = right.Clone();
        }

        if (leftRepaired || rightRepaired)
        {
            logger.LogDebug(
                "Eye frame repaired – left: {Left} (metric={LM:F4}, threshold={LT:F4}), " +
                "right: {Right} (metric={RM:F4}, threshold={RT:F4})",
                leftRepaired, result.LeftValue, result.LeftThreshold,
                rightRepaired, result.RightValue, result.RightThreshold);
        }

        // Emit periodic summary
        if (_logTimer.Elapsed.TotalSeconds >= SummaryIntervalSeconds)
        {
            var stats = GetDetector().GetStats();
            logger.LogInformation(
                "Eye corruption filter summary – total: {Total}, repaired left: {RL}, repaired right: {RR}, " +
                "threshold: {Threshold:F4} (adaptive: {Adaptive})",
                _totalFrames, _repairedLeft, _repairedRight,
                stats.CurrentThreshold, stats.AdaptiveEnabled);
            _logTimer.Restart();
        }

        transformed.Dispose();
        var repaired = new Mat();
        Cv2.Merge([left, right], repaired);
        left.Dispose();
        right.Dispose();

        return repaired;
    }

    private bool ProcessExpressions(ref float[] arKitExpressions)
    {
        if (arKitExpressions.Length < Utils.EyeRawExpressions)
            return false;

        const float mulV = 2.0f;
        const float mulY = 2.0f;

        var leftPitch = arKitExpressions[0] * mulY - mulY / 2;
        var leftYaw = arKitExpressions[1] * mulV - mulV / 2;
        var leftLid = 1 - arKitExpressions[2];

        var rightPitch = arKitExpressions[3] * mulY - mulY / 2;
        var rightYaw = arKitExpressions[4] * mulV - mulV / 2;
        var rightLid = 1 - arKitExpressions[5];

        var eyeY = (leftPitch * leftLid + rightPitch * rightLid) / (leftLid + rightLid);

        var leftEyeYawCorrected = rightYaw * (1 - leftLid) + leftYaw * leftLid;
        var rightEyeYawCorrected = leftYaw * (1 - rightLid) + rightYaw * rightLid;

        if (StabilizeEyes)
        {
            var rawConvergence = (rightEyeYawCorrected - leftEyeYawCorrected) / 2.0f;
            var convergence = Math.Max(rawConvergence, 0.0f); // We clamp the value here to avoid accidental divergence, as the model sometimes decides that's a thing

            var averagedYaw = (rightEyeYawCorrected + leftEyeYawCorrected) / 2.0f;

            leftEyeYawCorrected = averagedYaw - convergence;
            rightEyeYawCorrected = averagedYaw + convergence;
        }

        // [left pitch, left yaw, left lid...
        float[] convertedExpressions = new float[Utils.EyeRawExpressions];

        convertedExpressions[0] = rightEyeYawCorrected; // left pitch
        convertedExpressions[1] = eyeY;                   // left yaw
        convertedExpressions[2] = rightLid;               // left lid
        convertedExpressions[3] = leftEyeYawCorrected;  // right pitch
        convertedExpressions[4] = eyeY;                   // right yaw
        convertedExpressions[5] = leftLid;                // right lid

        arKitExpressions = convertedExpressions;

        return true;
    }


    public void Dispose()
    {
        TryDisposeObject(VideoSource);
        TryDisposeObject(ImageTransformer);
        TryDisposeObject(ImageConverter);
        TryDisposeObject(InferenceService);
        TryDisposeObject(Filter);
        TryDisposeObject(_imageCollector);
        _lastGoodLeft?.Dispose();
        _lastGoodRight?.Dispose();
        _lastGoodLeft = null;
        _lastGoodRight = null;
    }

    private void TryDisposeObject(object? obj)
    {
        (obj as IDisposable)?.Dispose();
    }
}
