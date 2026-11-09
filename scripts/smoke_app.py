"""Call the running UI through its API: python -m scripts.smoke_app [url]."""
import sys

from gradio_client import Client, handle_file

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
c = Client(url, verbose=False)
ref = open("samples/stt_01.txt", encoding="utf-8").read()
text, info, _ = c.predict(handle_file("samples/stt_01.wav"), "fa-turbo", ref, api_name="/run_stt")
print(text[:200], "\n", info)

para = "این یک آزمایش از رابط کاربری است. مسئول پروژه گفت که ۳۵ نفر در جلسه حضور داشتند."
wav, info = c.predict(handle_file("samples/voice_02.wav"), para, True, 0.5, api_name="/run_clone")
print(wav, info)
try:
    c.predict(handle_file("samples/voice_02.wav"), para, False, 0.5, api_name="/run_clone")
    print("CONSENT_NOT_ENFORCED")
except Exception as e:
    print("consent enforced:", str(e)[:80])
