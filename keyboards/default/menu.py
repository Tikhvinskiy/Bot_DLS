from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

menu = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="слабо"),
         KeyboardButton(text="больше"),
         KeyboardButton(text="норм"),
         KeyboardButton(text="много"),
         KeyboardButton(text="сюр")
        ],
           ],
    resize_keyboard=True,
    one_time_keyboard=True
)
