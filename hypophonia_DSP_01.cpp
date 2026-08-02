#include <Arduino.h>
#include <math.h>

#define FRAME_SIZE 512
#define HOP_SIZE 256
#define SAMPLE_RATE 860
#define NUM_MEL_BINS 12
#define NUM_FEATURES (NUM_MEL_BINS + 1)

volatile float ringBuffer[FRAME_SIZE];
volatile int ringWriteIndex = 0;
volatile int samplesSinceLastHop = 0;
volatile unsigned int totalSamples = 0;
volatile bool dataReadyToProcess = false;

float linearFrame[FRAME_SIZE];
float windowedFrame[FRAME_SIZE];
float windowCoeffs[FRAME_SIZE];

float mlInputTensor[NUM_FEATURES];
float melWeights[NUM_MEL_BINS][FRAME_SIZE / 2];
float dctMatrix[NUM_MEL_BINS][NUM_MEL_BINS];

// Pure C++ Radix-2 FFT (Replaces platform assembly dependency)
void computeFFT(float* real, float* imag, int n) {
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            float tempR = real[i]; real[i] = real[j]; real[j] = tempR;
            float tempI = imag[i]; imag[i] = imag[j]; imag[j] = tempI;
        }
        int k = n >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * M_PI / len;
        float wlenR = cosf(ang);
        float wlenI = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float wR = 1.0f;
            float wI = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                int u = i + k;
                int v = i + k + len / 2;
                float vrR = real[v] * wR - imag[v] * wI;
                float vrI = real[v] * wI + imag[v] * wR;
                real[v] = real[u] - vrR;
                imag[v] = imag[u] - vrI;
                real[u] += vrR;
                imag[u] += vrI;
                float nextWR = wR * wlenR - wI * wlenI;
                float nextWI = wR * wlenI + wI * wlenR;
                wR = nextWR;
                wI = nextWI;
            }
        }
    }
}

// Simulated ADC Signal (Software Signal Generator)
static int16_t readSimulatedADS1115() {
    static float time_sec = 0.0f;
    time_sec += 1.0f / SAMPLE_RATE;
    // 150 Hz tone (vocal pitch) + noise
    float signal_mV = 100.0f * sinf(2.0f * M_PI * 150.0f * time_sec);
    float noise_mV  = ((rand() % 200) - 100) / 10.0f;
    return (int16_t)((signal_mV + noise_mV) / 0.0078125f);
}

void initializeMelFilterbank() {
    auto hz2mel = [](float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); };
    auto mel2hz = [](float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); };

    float nyquist = SAMPLE_RATE / 2.0f;
    float melLow = hz2mel(80.0f);
    float melHigh = hz2mel(nyquist);
    float melStep = (melHigh - melLow) / (NUM_MEL_BINS + 1);

    int fftBins = FRAME_SIZE / 2;
    float binResolution = (float)SAMPLE_RATE / FRAME_SIZE;

    for (int m = 0; m < NUM_MEL_BINS; m++) {
        float melLeft = melLow + m * melStep;
        float melCenter = melLow + (m + 1) * melStep;
        float melRight = melLow + (m + 2) * melStep;

        float freqLeft = mel2hz(melLeft);
        float freqCenter = mel2hz(melCenter);
        float freqRight = mel2hz(melRight);

        int binLeft = round(freqLeft / binResolution);
        int binCenter = round(freqCenter / binResolution);
        int binRight = round(freqRight / binResolution);

        binLeft = constrain(binLeft, 0, fftBins - 1);
        binCenter = constrain(binCenter, 0, fftBins - 1);
        binRight = constrain(binRight, 0, fftBins - 1);

        for (int k = 0; k < fftBins; k++) {
            if (k >= binLeft && k <= binCenter) {
                melWeights[m][k] = (float)(k - binLeft) / max(1, binCenter - binLeft);
            } else if (k > binCenter && k <= binRight) {
                melWeights[m][k] = (float)(binRight - k) / max(1, binRight - binCenter);
            } else {
                melWeights[m][k] = 0.0f;
            }
        }
    }
}

void initializeDCTMatrix() {
    for (int i = 0; i < NUM_MEL_BINS; i++) {
        for (int j = 0; j < NUM_MEL_BINS; j++) {
            dctMatrix[i][j] = cosf((M_PI / NUM_MEL_BINS) * (j + 0.5f) * i);
        }
    }
}

void setup() {
    Serial.begin(115200);
    delay(500);

    Serial.println("\n--- [BOOT] Starting Pure C++ DSP Pipeline ---");

    // Precompute Hann Window coefficients
    for (int i = 0; i < FRAME_SIZE; i++) {
        windowCoeffs[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (FRAME_SIZE - 1)));
    }

    initializeMelFilterbank();
    initializeDCTMatrix();

    Serial.println("[SYSTEM] Setup complete. Streaming TENSORS:\n");
}

void processPhase1DSP() {
    dataReadyToProcess = false;

    int readIndex = ringWriteIndex;
    for (int i = 0; i < FRAME_SIZE; i++) {
        linearFrame[i] = ringBuffer[readIndex];
        readIndex = (readIndex + 1) % FRAME_SIZE;
    }

    // Apply Hann Window
    for (int i = 0; i < FRAME_SIZE; i++) {
        windowedFrame[i] = linearFrame[i] * windowCoeffs[i];
    }

    // RMS Energy Calculation (Tensor Element [0])
    float frameRMS = 0.0f;
    for (int i = 0; i < FRAME_SIZE; i++) {
        frameRMS += windowedFrame[i] * windowedFrame[i];
    }
    mlInputTensor[0] = sqrtf(frameRMS / FRAME_SIZE);

    // Prepare FFT Arrays
    float real[FRAME_SIZE];
    float imag[FRAME_SIZE];
    for (int i = 0; i < FRAME_SIZE; i++) {
        real[i] = windowedFrame[i];
        imag[i] = 0.0f;
    }

    computeFFT(real, imag, FRAME_SIZE);

    // Power Spectrum
    float powerSpectrum[FRAME_SIZE / 2];
    for (int i = 0; i < FRAME_SIZE / 2; i++) {
        powerSpectrum[i] = real[i] * real[i] + imag[i] * imag[i];
    }

    // Log-Mel Filterbank Energy
    float melEnergies[NUM_MEL_BINS];
    for (int m = 0; m < NUM_MEL_BINS; m++) {
        float energySum = 0.0f;
        for (int k = 0; k < FRAME_SIZE / 2; k++) {
            energySum += powerSpectrum[k] * melWeights[m][k];
        }
        melEnergies[m] = logf(energySum + 1e-6f);
    }

    // Discrete Cosine Transform (MFCC Extraction - Tensor Elements [1] to [12])
    for (int i = 0; i < NUM_MEL_BINS; i++) {
        float dctSum = 0.0f;
        for (int j = 0; j < NUM_MEL_BINS; j++) {
            dctSum += melEnergies[j] * dctMatrix[i][j];
        }
        mlInputTensor[i + 1] = dctSum;
    }

    // Output Input Tensor to Serial Terminal
    Serial.print("TENSOR_OUT:");
    for (int i = 0; i < NUM_FEATURES; i++) {
        Serial.print(mlInputTensor[i]);
        if (i < NUM_FEATURES - 1) Serial.print(",");
    }
    Serial.println();
}

void loop() {
    // Acquire continuous sample
    int16_t raw = readSimulatedADS1115();
    float sample_mV = (float)raw * 0.0078125f;

    ringBuffer[ringWriteIndex] = sample_mV;
    ringWriteIndex = (ringWriteIndex + 1) % FRAME_SIZE;
    totalSamples++;

    samplesSinceLastHop++;
    if (samplesSinceLastHop >= HOP_SIZE && totalSamples >= FRAME_SIZE) {
        dataReadyToProcess = true;
        samplesSinceLastHop = 0;
    }

    if (dataReadyToProcess) {
        processPhase1DSP();
    }

    // 860 Hz sampling period (~1163 microseconds per sample)
    delayMicroseconds(1163);
}