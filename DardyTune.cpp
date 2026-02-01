#include <JuceHeader.h>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <cstring>

static inline float clampf(float x, float lo, float hi) { return std::min(hi, std::max(lo, x)); }
static inline float midiToHz(float midi) { return 440.0f * std::pow(2.0f, (midi - 69.0f) / 12.0f); }
static inline float hzToMidi(float hz)
{
    if (hz <= 0.0f) return 0.0f;
    return 69.0f + 12.0f * std::log2(hz / 440.0f);
}

// ---- Scale quantizer (Major/Minor) ----
struct ScaleQuantizer
{
    int key = 0;   // 0=C..11=B
    int mode = 0;  // 0=Major, 1=Minor

    std::array<int, 12> majorMask {1,0,1,0,1,1,0,1,0,1,0,1};
    std::array<int, 12> minorMask {1,0,1,1,0,1,0,1,1,0,1,0};

    float quantizeMidi(float midi) const
    {
        const auto& mask = (mode == 0) ? majorMask : minorMask;
        int base = (int)std::round(midi);

        float best = (float)base;
        float bestDist = 1e9f;

        for (int d = -6; d <= 6; ++d)
        {
            int m = base + d;
            int pc = (m - key) % 12; if (pc < 0) pc += 12;

            if (mask[(size_t)pc])
            {
                float dist = std::abs((float)m - midi);
                if (dist < bestDist) { bestDist = dist; best = (float)m; }
            }
        }
        return best;
    }
};

// ---- Simple consonant/voicing detector (helps rap) ----
struct VoicingDetector
{
    float voicedConfidence(const float* x, int n) const
    {
        if (n < 64) return 0.0f;

        double e = 0.0;
        int zc = 0;
        float prev = x[0];

        for (int i = 0; i < n; ++i)
        {
            float s = x[i];
            e += (double)s * (double)s;
            if ((s >= 0) != (prev >= 0)) zc++;
            prev = s;
        }

        float rms = (float)std::sqrt(e / (double)n);
        float zcr = (float)zc / (float)n;

        float c1 = std::clamp((rms - 0.01f) / 0.08f, 0.0f, 1.0f);
        float c2 = 1.0f - std::clamp((zcr - 0.05f) / 0.25f, 0.0f, 1.0f);

        float conf = 0.65f * c1 + 0.35f * c2;
        return std::clamp(conf, 0.0f, 1.0f);
    }
};

// ---- Pitch detector (normalized autocorrelation) ----
struct PitchDetector
{
    void prepare(double sampleRate, int maxBlock)
    {
        sr = sampleRate;
        buf.resize((size_t)maxBlock);
    }

    std::pair<float, float> detect(const float* x, int n)
    {
        if (n < 128) return {0.0f, 0.0f};

        float mean = 0.0f;
        for (int i = 0; i < n; ++i) mean += x[i];
        mean /= (float)n;
        for (int i = 0; i < n; ++i) buf[(size_t)i] = x[i] - mean;

        const float fMin = 70.0f;
        const float fMax = 900.0f;
        int tauMin = std::max(2, (int)std::floor(sr / fMax));
        int tauMax = std::min(n - 2, (int)std::floor(sr / fMin));

        float bestScore = 0.0f;
        int bestTau = 0;

        for (int tau = tauMin; tau <= tauMax; ++tau)
        {
            double num = 0.0, den1 = 0.0, den2 = 0.0;
            for (int i = 0; i < n - tau; ++i)
            {
                float a = buf[(size_t)i];
                float b = buf[(size_t)(i + tau)];
                num  += (double)a * (double)b;
                den1 += (double)a * (double)a;
                den2 += (double)b * (double)b;
            }
            double den = std::sqrt(den1 * den2) + 1e-12;
            float score = (float)(num / den);

            if (score > bestScore) { bestScore = score; bestTau = tau; }
        }

        float conf = std::clamp((bestScore - 0.2f) / 0.8f, 0.0f, 1.0f);
        if (bestTau <= 0 || conf < 0.15f) return {0.0f, conf};

        float hz = (float)(sr / (double)bestTau);
        if (hz < fMin || hz > fMax) return {0.0f, 0.0f};

        return {hz, conf};
    }

    double sr = 44100.0;
    std::vector<float> buf;
};

// ---- Simple pitch shifter (prototype): JUCE dsp PitchShifter ----
struct SimplePitchShifter
{
    void prepare(double sampleRate, int maxBlock, int channels)
    {
        juce::dsp::ProcessSpec spec;
        spec.sampleRate = sampleRate;
        spec.maximumBlockSize = (juce::uint32)maxBlock;
        spec.numChannels = (juce::uint32)channels;

        shifter.prepare(spec);
        shifter.setLatencySamples(0);
    }

    void setRatio(float ratio)
    {
        shifter.setPitchShiftFactor(ratio);
    }

    void process(juce::AudioBuffer<float>& buffer)
    {
        juce::dsp::AudioBlock<float> block(buffer);
        juce::dsp::ProcessContextReplacing<float> ctx(block);
        shifter.process(ctx);
    }

    juce::dsp::PitchShifter<float> shifter;
};

//==============================================================================
class DardyTuneAudioProcessor : public juce::AudioProcessor
{
public:
    DardyTuneAudioProcessor()
        : AudioProcessor(BusesProperties()
            .withInput ("Input",  juce::AudioChannelSet::stereo(), true)
            .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
          apvts(*this, nullptr, "PARAMS", createParams())
    {}

    void prepareToPlay (double sampleRate, int samplesPerBlock) override
    {
        sr = sampleRate;
        mono.setSize(1, samplesPerBlock);
        pitch.prepare(sampleRate, samplesPerBlock);
        shifter.prepare(sampleRate, samplesPerBlock, getTotalNumOutputChannels());
        smoothedRatio = 1.0f;
    }

    void releaseResources() override {}

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
    {
        return layouts.getMainInputChannelSet() == layouts.getMainOutputChannelSet()
            && (layouts.getMainOutputChannelSet() == juce::AudioChannelSet::mono()
             || layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo());
    }

    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
    {
        juce::ScopedNoDenormals noDenormals;

        const int numCh = buffer.getNumChannels();
        const int n = buffer.getNumSamples();

        int key  = (int)*apvts.getRawParameterValue("key");
        int mode = (int)*apvts.getRawParameterValue("mode");
        float retuneMs = *apvts.getRawParameterValue("retune");
        float humanize = *apvts.getRawParameterValue("humanize");
        float protect  = *apvts.getRawParameterValue("protect");
        float mix      = *apvts.getRawParameterValue("mix");

        quant.key = key;
        quant.mode = mode;

        // mono analysis signal
        mono.setSize(1, n, false, false, true);
        float* m = mono.getWritePointer(0);
        for (int i = 0; i < n; ++i)
        {
            float s = 0.0f;
            for (int ch = 0; ch < numCh; ++ch) s += buffer.getReadPointer(ch)[i];
            m[i] = s / (float)numCh;
        }

        float voiced = voicing.voicedConfidence(m, n);
        if (voiced < protect)
            return; // consonants unshifted

        auto [hz, conf] = pitch.detect(m, n);
        if (hz <= 0.0f || conf < 0.2f)
            return;

        float midi = hzToMidi(hz);
        float targetMidi = quant.quantizeMidi(midi);

        // humanize keeps some drift
        targetMidi += (midi - targetMidi) * humanize;

        float ratio = clampf(midiToHz(targetMidi) / hz, 0.5f, 2.0f);

        // smoothing
        float tau = std::max(1.0f, retuneMs);
        float alpha = (float)std::exp(-(1000.0 / sr) * (float)n / tau);
        smoothedRatio = alpha * smoothedRatio + (1.0f - alpha) * ratio;

        // wet buffer
        wet.makeCopyOf(buffer, true);
        shifter.setRatio(smoothedRatio);
        shifter.process(wet);

        // mix wet/dry
        for (int ch = 0; ch < numCh; ++ch)
        {
            float* w = buffer.getWritePointer(ch);
            const float* wetPtr = wet.getReadPointer(ch);
            for (int i = 0; i < n; ++i)
                w[i] = (1.0f - mix) * w[i] + mix * wetPtr[i];
        }
    }

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override { return new juce::GenericAudioProcessorEditor(*this); }
    bool hasEditor() const override { return true; }

    //==============================================================================
    const juce::String getName() const override { return "DardyTune"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    //==============================================================================
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override
    {
        auto state = apvts.copyState();
        std::unique_ptr<juce::XmlElement> xml(state.createXml());
        copyXmlToBinary(*xml, destData);
    }

    void setStateInformation (const void* data, int sizeInBytes) override
    {
        std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
        if (xml && xml->hasTagName(apvts.state.getType()))
            apvts.replaceState(juce::ValueTree::fromXml(*xml));
    }

private:
    juce::AudioProcessorValueTreeState::ParameterLayout createParams()
    {
        std::vector<std::unique_ptr<juce::RangedAudioParameter>> p;

        p.push_back(std::make_unique<juce::AudioParameterChoice>(
            "key", "Key",
            juce::StringArray{"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"}, 0));

        p.push_back(std::make_unique<juce::AudioParameterChoice>(
            "mode", "Scale",
            juce::StringArray{"Major","Minor"}, 0));

        p.push_back(std::make_unique<juce::AudioParameterFloat>(
            "retune", "Retune (ms)",
            juce::NormalisableRange<float>(1.0f, 200.0f, 0.1f, 0.5f), 25.0f));

        p.push_back(std::make_unique<juce::AudioParameterFloat>(
            "humanize", "Humanize",
            juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 0.15f));

        p.push_back(std::make_unique<juce::AudioParameterFloat>(
            "protect", "Consonant Protect",
            juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 0.65f));

        p.push_back(std::make_unique<juce::AudioParameterFloat>(
            "mix", "Mix",
            juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 1.0f));

        return { p.begin(), p.end() };
    }

    double sr = 44100.0;

    juce::AudioProcessorValueTreeState apvts;

    juce::AudioBuffer<float> mono;
    juce::AudioBuffer<float> wet;

    ScaleQuantizer quant;
    VoicingDetector voicing;
    PitchDetector pitch;
    SimplePitchShifter shifter;

    float smoothedRatio = 1.0f;
};

//==============================================================================
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DardyTuneAudioProcessor();
}
