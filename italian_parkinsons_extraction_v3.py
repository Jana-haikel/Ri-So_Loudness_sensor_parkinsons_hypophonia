"""
Extract 13-element MFCC tensors from Italian Parkinson's Dataset
Manually compute MFCCs to avoid librosa shape conflicts
"""

import librosa
import numpy as np
from pathlib import Path
import json
from scipy.signal import windows

def rms_to_db(rms):
    """Convert RMS to dB SPL (20 µPa reference)"""
    ref = 1e-5
    db = 20 * np.log10(np.maximum(rms, 1e-10) / ref)
    return db

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def create_mel_filterbank(sr, n_fft, n_mels=12, f_min=80, f_max=3400):
    """Create triangular mel filterbank"""
    mel_f_min = hz_to_mel(f_min)
    mel_f_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_f_min, mel_f_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    bin_hz = sr / n_fft
    bin_points = hz_points / bin_hz
    
    filters = np.zeros((n_mels, n_fft // 2))
    for m in range(n_mels):
        left = int(np.floor(bin_points[m]))
        center = int(np.floor(bin_points[m + 1]))
        right = int(np.floor(bin_points[m + 2]))
        
        left = max(0, min(left, n_fft // 2 - 1))
        center = max(0, min(center, n_fft // 2 - 1))
        right = max(0, min(right, n_fft // 2 - 1))
        
        # Rising slope
        if center > left:
            for k in range(left, center):
                filters[m, k] = (k - left) / (center - left)
        # Falling slope
        if right > center:
            for k in range(center, right):
                filters[m, k] = (right - k) / (right - center)
    
    return filters

def extract_tensor(audio_path, n_fft=512, hop_length=256, n_mels=12):
    """
    Extract (13,) tensor per frame using manual MFCC computation.
    """
    try:
        # Load audio at native sample rate
        y, sr = librosa.load(audio_path, sr=None)
        
        if len(y) < n_fft:
            return None
        
        # Framing with Hann window
        window = windows.hann(n_fft)
        frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length)
        
        n_frames = frames.shape[1]
        tensors = np.zeros((n_frames, 13))
        
        # Mel filterbank
        mel_filters = create_mel_filterbank(sr, n_fft, n_mels=n_mels, f_min=80, f_max=3400)
        
        # DCT matrix (12x12)
        dct_matrix = np.zeros((n_mels, n_mels))
        for i in range(n_mels):
            for j in range(n_mels):
                dct_matrix[i, j] = np.cos((np.pi / n_mels) * (j + 0.5) * i)
        
        for t in range(n_frames):
            frame = frames[:, t] * window
            
            # RMS energy (converted to dB)
            rms = np.sqrt(np.mean(frame ** 2))
            tensors[t, 0] = rms_to_db(rms)
            
            # FFT
            fft = np.fft.rfft(frame)
            power = np.abs(fft) ** 2
            
            # Mel filterbank
            mel_power = np.dot(mel_filters, power[:n_fft // 2])
            mel_power = np.maximum(mel_power, 1e-6)  # Avoid log(0)
            mel_log = np.log(mel_power)
            
            # DCT
            mfcc = np.dot(dct_matrix, mel_log)
            tensors[t, 1:] = mfcc
        
        return tensors
    
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None

def detect_label(audio_path):
    """Detect label from folder path."""
    full_path = str(audio_path).lower()
    
    # Check if "28 people" folder is in the path
    if "28 people" in full_path or "28people" in full_path:
        return "PARKINSONS"
    elif "healthy" in full_path or "control" in full_path or "young" in full_path or "elderly" in full_path:
        return "HEALTHY"
    else:
        return "UNKNOWN"

def main():
    dataset_path = Path(r"/home/janahaikel/Downloads/Italian%20Parkinson%27s%20Voice%20and%20speech")
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        print("   Update the path in the script to your download location")
        return
    
    audio_files = list(dataset_path.glob("**/*.wav"))
    print(f"Found {len(audio_files)} audio files\n")
    
    if not audio_files:
        print("❌ No .wav files found.")
        return
    
    healthy_files = [f for f in audio_files if "healthy" in str(f).lower() or "control" in str(f).lower()]
    parkinsons_files = [f for f in audio_files if "28 people" in str(f).lower()]
    
    print(f"Healthy Control files: {len(healthy_files)}")
    print(f"Parkinson's files: {len(parkinsons_files)}\n")
    
    if parkinsons_files:
        print(f"Sample Parkinson's file: {parkinsons_files[0]}\n")
    if healthy_files:
        print(f"Sample Healthy file: {healthy_files[0]}\n")
    
    validation_healthy = healthy_files[:20]
    validation_parkinsons = parkinsons_files[:20]
    validation_samples = validation_healthy + validation_parkinsons
    
    results = []
    sample_count = 0
    
    print("Extracting tensors...\n")
    
    for audio_file in validation_samples:
        sample_count += 1
        label = detect_label(audio_file)
        
        print(f"[{sample_count}/10] {label:12} | {audio_file.name}")
        
        tensors = extract_tensor(audio_file)
        if tensors is not None:
            rms_values = tensors[:, 0]
            mfcc_values = tensors[:, 1:]
            
            results.append({
                "filename": audio_file.name,
                "label": label,
                "n_frames": tensors.shape[0],
                "rms_mean": float(np.mean(rms_values)),
                "rms_std": float(np.std(rms_values)),
                "rms_min": float(np.min(rms_values)),
                "rms_max": float(np.max(rms_values)),
                "mfcc_mean": float(np.mean(mfcc_values)),
            })
            
            np.save(f"sample_{sample_count:02d}_{label}.npy", tensors)
            print(f"           ✓ {tensors.shape[0]} frames extracted\n")
        else:
            print(f"           ⚠️  Skipped\n")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY — Comparing HEALTHY vs PARKINSONS")
    print("="*80)
    
    healthy_results = [r for r in results if r["label"] == "HEALTHY"]
    parkinsons_results = [r for r in results if r["label"] == "PARKINSONS"]
    
    print(f"\nHEALTHY CONTROL ({len(healthy_results)} samples):")
    print("-" * 80)
    for r in healthy_results:
        print(f"{r['filename']:40} | RMS: {r['rms_mean']:6.2f} ± {r['rms_std']:5.2f}")
    
    if healthy_results:
        avg_healthy_rms = np.mean([r["rms_mean"] for r in healthy_results])
        print(f"{'Average (HEALTHY)':40} | RMS: {avg_healthy_rms:6.2f}")
    
    print(f"\nPARKINSON'S DISEASE ({len(parkinsons_results)} samples):")
    print("-" * 80)
    for r in parkinsons_results:
        print(f"{r['filename']:40} | RMS: {r['rms_mean']:6.2f} ± {r['rms_std']:5.2f}")
    
    if parkinsons_results:
        avg_parkinsons_rms = np.mean([r["rms_mean"] for r in parkinsons_results])
        print(f"{'Average (PARKINSONS)':40} | RMS: {avg_parkinsons_rms:6.2f}")
    
    if healthy_results and parkinsons_results:
        ratio = avg_healthy_rms / avg_parkinsons_rms
        print(f"\n💡 INSIGHT: Healthy voices are ~{ratio:.1f}x louder (higher RMS)")
        print("   This is hypophonia — the core biomarker for Parkinson's detection!")
    
    with open("extraction_metadata.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ Done. Saved .npy files and extraction_metadata.json")
    print("="*80)

if __name__ == "__main__":
    main()
