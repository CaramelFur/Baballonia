using Microsoft.Extensions.Logging;
using OpenCvSharp;
using System;

namespace Baballonia.Services.Inference;

public class DualImageTransformer : IImageTransformer, IDisposable
{
    public ImageTransformer LeftTransformer = new();
    public ImageTransformer RightTransformer = new();

    public bool RepairEnabled { get; private set; } = true;
    public double RepairThreshold { get; private set; } = 0.022669;
    public int MaxConsecutiveRepairs { get; private set; } = 3;

    private readonly ILogger<DualImageTransformer>? _logger;

    private FastCorruptionDetector.FastCorruptionDetector _leftDetector;
    private FastCorruptionDetector.FastCorruptionDetector _rightDetector;

    private Mat? _prevGoodLeft;
    private Mat? _prevGoodRight;
    private int _leftConsecutiveRepairs;
    private int _rightConsecutiveRepairs;

    private int _leftRepairCount;
    private int _rightRepairCount;
    private int _leftForceAcceptCount;
    private int _rightForceAcceptCount;
    private DateTime _lastLogTime = DateTime.MinValue;
    private static readonly TimeSpan LogInterval = TimeSpan.FromSeconds(30);

    public DualImageTransformer(ILogger<DualImageTransformer>? logger = null)
    {
        _logger = logger;
        _leftDetector = new FastCorruptionDetector.FastCorruptionDetector(RepairThreshold);
        _rightDetector = new FastCorruptionDetector.FastCorruptionDetector(RepairThreshold);
    }

    public void SetRepairConfig(bool enabled, double threshold, int maxConsecutiveRepairs)
    {
        RepairEnabled = enabled;
        if (Math.Abs(threshold - RepairThreshold) > 1e-10)
        {
            RepairThreshold = threshold;
            _leftDetector = new FastCorruptionDetector.FastCorruptionDetector(threshold);
            _rightDetector = new FastCorruptionDetector.FastCorruptionDetector(threshold);
        }
        MaxConsecutiveRepairs = maxConsecutiveRepairs;
    }

    public Mat? Apply(Mat image)
    {
        // Assuming the frame is wide enough to be split in half
        var width = image.Width;
        var height = image.Height;

        // Split the frame into left and right halves
        var leftHalf = new Rect(0, 0, width / 2, height);
        var rightHalf = new Rect(width / 2, 0, width / 2, height);

        // Create ROIs for left and right eyes
        using var leftRoi = new Mat(image, leftHalf);
        using var rightRoi = new Mat(image, rightHalf);

        // transform both simultaneously with same settings
        var leftTransformed = LeftTransformer.Apply(leftRoi);
        var rightTransformed = RightTransformer.Apply(rightRoi);
        if (leftTransformed == null || rightTransformed == null)
        {
            leftTransformed?.Dispose();
            rightTransformed?.Dispose();
            return null;
        }

        if (RepairEnabled)
        {
            leftTransformed = ApplyPerEyeRepair(leftTransformed, ref _prevGoodLeft, ref _leftConsecutiveRepairs, _leftDetector, isLeft: true);
            rightTransformed = ApplyPerEyeRepair(rightTransformed, ref _prevGoodRight, ref _rightConsecutiveRepairs, _rightDetector, isLeft: false);
        }

        var combined = new Mat();
        Cv2.Merge([leftTransformed, rightTransformed], combined);

        leftTransformed.Dispose();
        rightTransformed.Dispose();

        LogPeriodicStats();

        return combined;
    }

    private Mat ApplyPerEyeRepair(
        Mat current,
        ref Mat? prevGood,
        ref int consecutiveRepairs,
        FastCorruptionDetector.FastCorruptionDetector detector,
        bool isLeft)
    {
        var (isCorrupted, metricValue, thresholdUsed) = detector.IsCorrupted(current);
        var eyeName = isLeft ? "left" : "right";

        if (!isCorrupted)
        {
            prevGood?.Dispose();
            prevGood = current.Clone();
            consecutiveRepairs = 0;
            return current;
        }

        // Frame is corrupted – substitute previous good if available and within limit
        if (prevGood != null && consecutiveRepairs < MaxConsecutiveRepairs)
        {
            consecutiveRepairs++;
            if (isLeft) _leftRepairCount++;
            else _rightRepairCount++;

            _logger?.LogDebug(
                "Eye frame repair ({Eye}): substituting previous good frame (metric={Metric:F6}, threshold={Threshold:F6}, consecutive={Count}/{Max})",
                eyeName, metricValue, thresholdUsed, consecutiveRepairs, MaxConsecutiveRepairs);

            current.Dispose();
            return prevGood.Clone();
        }

        // Force accept: no previous good frame exists, or consecutive repair limit reached
        if (prevGood != null)
        {
            if (isLeft) _leftForceAcceptCount++;
            else _rightForceAcceptCount++;

            _logger?.LogInformation(
                "Eye frame repair ({Eye}): forcing accept after {MaxRepairs} consecutive repairs (metric={Metric:F6}, threshold={Threshold:F6})",
                eyeName, MaxConsecutiveRepairs, metricValue, thresholdUsed);
        }

        prevGood?.Dispose();
        prevGood = current.Clone();
        consecutiveRepairs = 0;
        return current;
    }

    private void LogPeriodicStats()
    {
        if (_logger == null)
            return;

        var now = DateTime.UtcNow;
        if (now - _lastLogTime < LogInterval)
            return;

        _lastLogTime = now;

        if (_leftRepairCount > 0 || _rightRepairCount > 0 || _leftForceAcceptCount > 0 || _rightForceAcceptCount > 0)
        {
            _logger.LogInformation(
                "Eye frame repair stats (last {Interval}s): left repairs={LeftRepairs}, right repairs={RightRepairs}, left forced accepts={LeftForced}, right forced accepts={RightForced}",
                (int)LogInterval.TotalSeconds, _leftRepairCount, _rightRepairCount, _leftForceAcceptCount, _rightForceAcceptCount);

            _leftRepairCount = 0;
            _rightRepairCount = 0;
            _leftForceAcceptCount = 0;
            _rightForceAcceptCount = 0;
        }
    }

    public void Dispose()
    {
        _prevGoodLeft?.Dispose();
        _prevGoodLeft = null;
        _prevGoodRight?.Dispose();
        _prevGoodRight = null;
    }
}
