import sys
import pyttsx3

# com.apple.voice.enhanced.en-GB.Stephanie
# com.apple.voice.premium.en-GB.Malcolm
# com.apple.voice.enhanced.zh-CN.Tingting


def init_engine():
    engine = pyttsx3.init()
    voice_id = "com.apple.voice.premium.en-GB.Malcolm"
    engine.setProperty("voice", voice_id)
    engine.setProperty("rate", 153)

    return engine


def say(command):
    engine.say(command)
    engine.runAndWait()  # blocks


engine = init_engine()
say(str(sys.argv[1]))
