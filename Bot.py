import telebot
import subprocess
from telebot.types import Message
import requests
from test import main
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
API_TOKEN = '1890866967:AAHnrZwDBQsTHGKZs4WjY53_YmZvFYvkpFM'


bot = telebot.TeleBot(API_TOKEN)


@bot.message_handler(commands=['start'])
def handel(message):
    bot.send_message(
        message.chat.id, "Bienvenido al entorno de reconociemiento de emociones.")
    bot.send_message(message.chat.id, "Por favor solo utilizar notas de voz.")


@bot.message_handler(content_types=['text'])
def handel(message):
    bot.reply_to(message, "Por favor solo utilizar notas de voz")


@bot.message_handler(content_types=['voice'])
def handel(message):

    fileID = message.voice.file_id
    file = bot.get_file(fileID)
    down_file = bot.download_file(file.file_path)
    with open('test.ogg', 'wb') as f:
        f.write(down_file)

    src_filename = 'test.ogg'
    dest_filename = 'test.wav'

    process = subprocess.run(
        ['C:\\\FFmpeg\\bin\\ffmpeg.exe', '-i', src_filename, dest_filename, '-y'])
    if process.returncode != 0:
        raise Exception("Error")

    result = main()
    bot.send_message(message.chat.id, result)


bot.polling(none_stop=True)
