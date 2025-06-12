"""Simple CLI entry point for running speaker diarization locally."""
import argparse
import json
from pprint import pprint

from nssd.speaker_diarization import SpeakerDiarization


def _build_config(args):
    # load configurations
    config = json.loads(open(args.config, 'r').read())
    return config


def main(args):
    """Run diarization on a WAV file."""
    config = _build_config(args)
    pprint(config)
    speaker_diarization = SpeakerDiarization(**config)

    run_result = speaker_diarization.run([args.input_pcm])
    _, sel_tuples = run_result.export_nssd_data()
    pprint(sel_tuples[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run speaker diarization on a WAV file.")
    parser.add_argument('--config',
                        dest='config',
                        type=str,
                        help='configuration file for speaker diarization.',
                        default='configs/speaker_diarization.conf')
    parser.add_argument('input_pcm',
                        type=str,
                        help='Path to input WAV file')
    main(parser.parse_args())
