import asyncio

from model.model import Transformation
import torch
from aiogram import types
from aiogram.dispatcher.filters.builtin import CommandStart, Command, Text
from aiogram.types import ContentType, Message
from pathlib import Path

from keyboards.default import menu
from loader import dp, bot

start = False
power = False
preview = True


@dp.message_handler(CommandStart())
async def bot_start(message: types.Message):
    global start
    global preview

    if preview:
        await message.answer(f"Привет, {message.from_user.full_name}.\n"
                             f"Я бот по переносу стиля. Смотри пример.")
        media = types.MediaGroup()
        media.attach_photo(types.InputFile('./photos/mask1.jpg'), 'До стиля')
        media.attach_photo(types.InputFile('./photos/picasso.jpg'), 'Стиль')
        media.attach_photo(types.InputFile('./photos/mask2.jpg'), 'После переноса стиля')
        await message.answer_media_group(media=media)

    await message.answer(f"Теперь пришли мне две фотографии в формате jpg.\n"
                         f"Первая которую будем стилизовать, вторая сам стиль.\n"
                         f"\n"
                         f"(жми сюда /change_style если уже загружал фотки и хочешь изменить силу наложения стиля)")
    start = True
    preview = False


@dp.message_handler(content_types=ContentType.PHOTO)
async def get_photo(message: Message):
    global start
    global power

    chat_id = message.chat.id
    path_to_download = Path().joinpath("photos")
    path_to_download.mkdir(parents=True, exist_ok=True)
    if start:
        path_to_download = path_to_download.joinpath(f'content{chat_id}.jpg')
        await message.photo[-1].download(destination=path_to_download)
        await message.answer(f"Фото для обработки было сохранено в путь: {path_to_download}")
        await message.answer(f"Теперь загрузи картинку стиля")
        start = False
    else:
        path_to_download = path_to_download.joinpath(f'style{chat_id}.jpg')
        await message.photo[-1].download(destination=path_to_download)
        await message.answer(f"Фото стиля было сохранено в путь: {path_to_download}")
        await message.answer(f"Теперь выбери силу стиля (используй клавиатуру бота)", reply_markup=menu)
        power = True


@dp.message_handler(Command('change_style'))
async def get_style2(message: Message):
    global power
    await message.answer(f"Выбери силу стиля (используй клавиатуру бота)", reply_markup=menu)
    power = True


@dp.message_handler(Text(equals=["слабо", "больше", "норм", "много", "сюр"]))
async def get_style1(message: Message):
    await asyncio.sleep(1)
    if power:
        await message.answer(f"Вы выбрали силу переноса стиля '{message.text}'.")
        style_power = 3
        if message.text == "слабо":
            style_power = 1.3
        elif message.text == "больше":
            style_power = 2
        elif message.text == "много":
            style_power = 4
        elif message.text == "сюр":
            style_power = 5

        kernel = 200
        epoch = 200
        device = 'cpu'
        # нестабильно CUDA на сервере работала, отключил
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)
        # if str(device) == "cpu":
        #     await message.answer(f"Видеокарта не обнаружена, расчет ведется на CPU.\n"
        #                          f"Придется подождать...")
        #     kernel = 200
        #     epoch = 200
        # elif str(device) == "cuda":
        #     await message.answer(f"Видеокарта обнаружена, расчет ведется на GPU.")
        #     kernel = 300
        #     epoch = 200
        #     torch.cuda.empty_cache()

        chat_id = message.chat.id
        print(chat_id)
        model = Transformation(style_power, kernel, chat_id, epoch, device)
        content = Path().joinpath(f"./photos/content{chat_id}.jpg")
        style = Path().joinpath(f"./photos/style{chat_id}.jpg")
        try:
            model.processing(content, style)
        except Exception:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            await message.answer(f'Что-то пошло не так, попробуй еще')

        with open(f'./photos/out{chat_id}.jpg', 'rb') as photo:
            await bot.send_photo(message.chat.id, photo,
                                 caption=f'Результат применения стиля.\n')
            await message.answer(f'Хочешь попробовать еще c новыми фотками /start\n'
                                 f'Хочешь изменить силу наложения стиля загруженной фотки /change_style\n'
                                 f'Хочешь удалить свои картинки с сервера и уйти /exit')
    else:
        return


@dp.message_handler(Command('exit'))
async def go_out(message: Message):
    chat_id = message.chat.id
    try:
        for file in [f'./photos/content{chat_id}.jpg', f'./photos/out{chat_id}.jpg', f'./photos/style{chat_id}.jpg']:
            Path(file).unlink()
            await message.answer(f"{file} - удален")

    except Exception as er:
        await message.answer(f"{er}")

    await message.answer("Пока!")
