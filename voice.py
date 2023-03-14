import sys
import pyttsx3

# com.apple.voice.enhanced.en-GB.Stephanie
# com.apple.voice.premium.en-GB.Malcolm
# com.apple.voice.premium.zh-CN.Lili
# com.apple.voice.premium.zh-CN.Lilian


def init_engine():
    engine = pyttsx3.init()
    voice_id = "com.apple.voice.premium.en-GB.Malcolm"
    engine.setProperty("voice", voice_id)
    engine.setProperty("rate", 165)

    return engine


def say(s):
    engine.say(s)
    engine.runAndWait()  # blocks


engine = init_engine()
say(str(sys.argv[1]))
