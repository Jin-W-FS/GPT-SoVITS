import argparse
import os, sys
import soundfile as sf

sys.path.append(os.getcwd())
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

i18n = I18nAuto()

def synthesize(GPT_model_path, SoVITS_model_path, ref_audio_path, ref_text_path, ref_language, target_text_path, target_language, output_path):
    # Read reference text
    if os.path.exists(ref_text_path):
        with open(ref_text_path, 'r', encoding='utf-8') as file:
            ref_text = file.read()
    else:
        ref_text = ref_text_path

    # Read target text
    if os.path.exists(target_text_path):
        with open(target_text_path, 'r', encoding='utf-8') as file:
            target_text = file.read()
    else:
        target_text = target_text_path

    # Change model weights
    change_gpt_weights(gpt_path=GPT_model_path)
    change_sovits_weights(sovits_path=SoVITS_model_path)

    # Synthesize audio
    synthesis_result = get_tts_wav(ref_wav_path=ref_audio_path, 
                                   prompt_text=ref_text, 
                                   prompt_language=i18n(ref_language), 
                                   text=target_text, 
                                   text_language=i18n(target_language), top_p=1, temperature=1)
    
    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        output_wav_path = os.path.join(output_path, "output.wav")
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")

def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")
    parser.add_argument('--gpt_model', default="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", help="Path to the GPT model file")
    parser.add_argument('--sovits_model', default="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth", help="Path to the SoVITS model file")
    parser.add_argument('--ref_audio', required=True, help="Path to the reference audio file")
    parser.add_argument('--ref_text', required=True, help="Path to the reference text file")
    parser.add_argument('--ref_language', default="中文", choices=["中文", "英文", "日文"], help="Language of the reference audio")
    parser.add_argument('--target_text', required=True, help="Path to the target text file")
    parser.add_argument('--target_language', default="中文", choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"], help="Language of the target text")
    parser.add_argument('--output_path', default="output/", help="Path to the output directory")

    args = parser.parse_args()

    synthesize(args.gpt_model, args.sovits_model, args.ref_audio, args.ref_text, args.ref_language, args.target_text, args.target_language, args.output_path)

if __name__ == '__main__':
    main()

