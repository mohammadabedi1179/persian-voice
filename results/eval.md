# Evaluation results

## Speech-to-text

| model | corpus WER | CER | CER (no space) | RTF | VRAM GB |
|---|---|---|---|---|---|
| fa-turbo | 8.9% | 2.8% | 2.8% | 0.044 | 2.11 |
| base-turbo | 18.0% | 5.4% | 5.7% | 0.044 | 2.11 |

Corpus: 300 FLEURS fa test utterances (78.2 min).

| clip | model | seconds | WER | CER |
|---|---|---|---|---|
| stt_01.wav (43.04 s) | fa-turbo | 1.9 | 3.3% | 0.4% |
| stt_02.wav (51.72 s) | fa-turbo | 1.8 | 7.6% | 2.2% |
| stt_03.wav (40.76 s) | fa-turbo | 1.48 | 2.8% | 1.1% |
| stt_01.wav (43.04 s) | base-turbo | 1.27 | 18.3% | 4.1% |
| stt_02.wav (51.72 s) | base-turbo | 1.82 | 16.5% | 4.3% |
| stt_03.wav (40.76 s) | base-turbo | 1.54 | 5.6% | 1.9% |

## Voice cloning, before fixes

Model `parsvoice-xtts`, 12 generations, device cuda, peak VRAM 3.67 GB.

- mean speaker similarity (reference vs clone): **0.65**
- between reference clips (FLEURS has no speaker ids, so some may be one person): 0.28
- round-trip WER / CER (clone -> STT): 39.5% / 29.0%
- mean RTF: 0.316

| voice | paragraph | audio s | RTF | similarity | round-trip CER |
|---|---|---|---|---|---|
| voice_01.wav | 1 | 19.31 | 0.327 | 0.76 | 16.6% |
| voice_01.wav | 2 | 15.54 | 0.303 | 0.72 | 8.9% |
| voice_01.wav | 3 | 15.12 | 0.304 | 0.74 | 5.2% |
| voice_02.wav | 1 | 22.05 | 0.304 | 0.72 | 9.7% |
| voice_02.wav | 2 | 15.22 | 0.304 | 0.72 | 3.4% |
| voice_02.wav | 3 | 13.23 | 0.304 | 0.73 | 5.9% |
| voice_03.wav | 1 | 25.9 | 0.302 | 0.55 | 178.3% |
| voice_03.wav | 2 | 19.07 | 0.303 | 0.46 | 38.4% |
| voice_03.wav | 3 | 16.75 | 0.303 | 0.53 | 47.4% |
| voice_04.wav | 1 | 16.98 | 0.307 | 0.62 | 9.7% |
| voice_04.wav | 2 | 15.08 | 0.35 | 0.65 | 14.4% |
| voice_04.wav | 3 | 14.94 | 0.384 | 0.58 | 10.4% |

## Voice cloning, final settings, one take

Model `parsvoice-xtts`, 12 generations, device cpu, peak VRAM 0 GB.

- mean speaker similarity (reference vs clone): **0.69**
- between reference clips (FLEURS has no speaker ids, so some may be one person): 0.28
- round-trip WER / CER (clone -> STT): 10.6% / 3.4%
- mean RTF: 1.074
- held out from tuning (6 clips: voice_02 or paragraph 3): WER / CER 13.6% / 4.0%, similarity 0.70

| voice | paragraph | audio s | RTF | similarity | round-trip CER |
|---|---|---|---|---|---|
| voice_01.wav | 1 | 16.62 | 1.103 | 0.71 | 5.1% |
| voice_01.wav | 2 | 14.61 | 1.142 | 0.70 | 2.7% |
| voice_01.wav | 3 | 15.77 | 1.17 | 0.71 | 3.7% |
| voice_02.wav | 1 | 16.37 | 1.049 | 0.73 | 5.7% |
| voice_02.wav | 2 | 12.48 | 1.057 | 0.76 | 2.7% |
| voice_02.wav | 3 | 12.89 | 1.087 | 0.73 | 3.0% |
| voice_03.wav | 1 | 16.09 | 1.064 | 0.71 | 0.6% |
| voice_03.wav | 2 | 11.79 | 1.035 | 0.70 | 0.7% |
| voice_03.wav | 3 | 11.69 | 1.039 | 0.65 | 5.9% |
| voice_04.wav | 1 | 20.69 | 1.027 | 0.63 | 5.7% |
| voice_04.wav | 2 | 15.44 | 1.012 | 0.65 | 1.4% |
| voice_04.wav | 3 | 13.23 | 1.1 | 0.66 | 3.0% |

## Voice cloning, final settings, up to 3 takes per sentence

Model `parsvoice-xtts`, 12 generations, device cpu, peak VRAM 0 GB.

- mean speaker similarity (reference vs clone): **0.69**
- between reference clips (FLEURS has no speaker ids, so some may be one person): 0.28
- round-trip WER / CER (clone -> STT): 12.7% / 4.7%
- mean RTF: 3.856
- held out from tuning (6 clips: voice_02 or paragraph 3): WER / CER 14.5% / 6.0%, similarity 0.71

| voice | paragraph | audio s | RTF | similarity | round-trip CER |
|---|---|---|---|---|---|
| voice_01.wav | 1 | 15.22 | 4.278 | 0.70 | 5.1% |
| voice_01.wav | 2 | 13.31 | 2.662 | 0.69 | 2.1% |
| voice_01.wav | 3 | 14.65 | 4.443 | 0.69 | 3.0% |
| voice_02.wav | 1 | 16.01 | 2.773 | 0.73 | 5.1% |
| voice_02.wav | 2 | 13.31 | 3.508 | 0.75 | 1.4% |
| voice_02.wav | 3 | 13.49 | 5.078 | 0.74 | 8.2% |
| voice_03.wav | 1 | 14.61 | 6.208 | 0.71 | 3.4% |
| voice_03.wav | 2 | 11.45 | 2.095 | 0.69 | 2.1% |
| voice_03.wav | 3 | 12.52 | 3.734 | 0.68 | 14.8% |
| voice_04.wav | 1 | 15.59 | 5.518 | 0.60 | 4.6% |
| voice_04.wav | 2 | 16.05 | 3.101 | 0.65 | 2.7% |
| voice_04.wav | 3 | 14.85 | 2.871 | 0.68 | 3.7% |
