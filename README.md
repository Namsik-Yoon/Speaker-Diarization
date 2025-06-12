# Speaker Diarization

This project contains code for training speaker embedding models and running a simple diarization pipeline. The original implementation relied on the NSML platform, but the scripts have been adapted for normal Python execution.

## Running diarization

```
python SpeakerDiarization/main.py path/to/audio.wav --config SpeakerDiarization/configs/speaker_diarization.conf
```

## Training SpeakerNet

```
cd SpeakerNetTrainer
python trainSpeakerNet.py --train_list <train_list> --test_list <test_list> --save_path exp
```

The training script will create `exp/model` and `exp/result` directories to store checkpoints and evaluation results.
