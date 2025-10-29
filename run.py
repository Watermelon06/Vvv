import os
import re
import time
import json
import random
import asyncio
import logging
import aiohttp
import aiosqlite
from pydub import AudioSegment
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import Filter
from aiogram.filters.command import Command
from aiogram.filters.logic import or_f
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import InlineKeyboardButton, FSInputFile, CallbackQuery, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from sqlalchemy import BigInteger, select, update, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import AsyncAttrs, async_sessionmaker, create_async_engine
from sqlalchemy import func
from rev_ai import apiclient
from moviepy.editor import VideoFileClip

ADMINS = [7281169403]
API_TOKEN = '6601937260:AAHHoZOntirOMryKbBsws5ukO9OqJpzyTuo'
logging.basicConfig(level=logging.INFO)
bot = Bot(token=API_TOKEN)
dp = Dispatcher()
tokens = ['02j2kY_UvdoL7WjGdXSyQ9MqLr9A-4oGoR6Z2JZt6BUh91471ctMr1FUD7oWGI-Kahzhoq6VZ7ZpLf4vLI4dyxYvvbHec', '02LIXJoYtq24oWGQ_-8Outb3c9C9kuOZZOF-Eg99NmlZ_-IDT5_8p0PI9OSq6_GtqSenZz0tlwSXcAYFWuS51L8O6lMBg', '02j61_VBQhxPGg21wPHh3hP_knvV2t2oshg3yc1Ci7SUfuMYSm-afV_VKUBq4lia530468ho0nPRyCYcNg92YhhR2kDg8', '02PIllMqo68kATt3yTO2fgQaX6GR93p5sfWp3MjHebCdWA82rSGKGeBc_emzWm10raufa_iEIq81d_r-5RycCMkN8bX-8']

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
ogg_path = os.path.join(DATA_DIR, 'add.ogg')
db_path = os.path.join(DATA_DIR, 'db.sqlite3')
BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')



# Создаем папки, если они не существуют
os.makedirs(os.path.dirname(db_path), exist_ok=True)
# Создаем файл базы данных, если он не существует
if not os.path.exists(db_path):
    with open(db_path, 'w'):
        pass

DATABASE_URL = f'sqlite+aiosqlite:///{db_path}'

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = async_sessionmaker(engine)


class AdminProtect(Filter):
    def __init__(self):
        self.admins = ADMINS

    async def __call__(self, message: Message):
        return message.from_user.id in self.admins


class Newsletter(StatesGroup):
    message = State()


class Add_tokens(StatesGroup):
    message = State()


class Base(AsyncAttrs, DeclarativeBase):
    pass


class User(Base):
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(primary_key=True)
    tg_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    username: Mapped[str] = mapped_column(default='Noname', nullable=False)
    time: Mapped[int] = mapped_column(default=1800, nullable=False)
    pro: Mapped[int] = mapped_column(default=0, nullable=False)



async def async_main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def create_user(tg_id, username):
    async with async_session() as session:
        user = await session.scalar(select(User).where(User.tg_id == tg_id))
        if not user:
            session.add(User(tg_id=tg_id, username=username))
        else:
            await session.execute(update(User).where(User.tg_id == tg_id).values(username=username))
        await session.commit()


async def add_usage(tg_id, usage):
    async with async_session() as session:
        user = await session.scalar(select(User).where(User.tg_id == tg_id))
        if user.pro==1:
            pass
        else:
            if user.time < usage:
                return True

            user.time -= usage
            await session.commit()


async def get_users():
    async with async_session() as session:
        users = await session.scalars(select(User))
        return users
    

async def get_user(tg_id):
    async with async_session() as session:
        return await session.scalar(select(User).where(User.tg_id == tg_id))


async def get_time(tg_id):
    user = await get_user(tg_id)
    return user.time if user else None


async def check_pro(tg_id):
    user = await get_user(tg_id)
    return int(user.pro) if user else None
    

def get_duration_pydub(file_path):
    audio = AudioSegment.from_file(file_path)
    return int(len(audio) / 1000.0)


@dp.message(Command('start'))
async def start(message: Message):
    await message.answer(
        'Нет Telegram premium🚀? Не беда! Просто Отправь мне аудио или видео(кружочек) для перевода в текстовый вариант 📝 😉', 
        reply_markup=ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text='Меню')]], resize_keyboard=True))
    await create_user(message.from_user.id, str(message.from_user.username))


@dp.message(or_f(Command("menu"), F.text.lower() == 'меню'))
async def menu_or_balance_handler(message: Message):
    user = await get_user(message.from_user.id)

    if not user:
        await message.answer("Не удалось найти ваш профиль. Нажмите /start для регистрации.")
        return

    pro = int(user.pro)
    text = "💼 Главное меню\n"
    if pro:
        text += "\n🟢 У вас активен PRO-доступ. Количество минут не ограничено"
    else:
        times = user.time
        text += f"\n🔴 У вас обычный доступ.\nУ вас осталось в этом месяце: {times//60} мин. {times%60} сек. \n\n❗Чтобы получить PRO — нажмите /buy\n\n В бесплатной версии дается 30 минут каждый месяц 1-го числа. 🤑 PRO доступ - доступ к переводу голосовых в текст без ограничений по времени и длительности аудио. Стоимость 40 р/мес"

    await message.answer(text)


@dp.message(Command('buy'))
async def buy_pro(message: Message):
    await message.answer('Пока что функция быстрой оплаты в разработке, для оплаты напиши мне в лс @vikwo2pps')

async def find_user_by_id_or_username(identifier: str):
    async with async_session() as session:
        # Если передан числовой ID
        if identifier.isdigit():
            user = await session.scalar(select(User).where(User.tg_id == int(identifier)))
        else:
            # Поиск по username (без @)
            username = identifier.lstrip('@')
            user = await session.scalar(select(User).where(User.username == username))
        return user


@dp.message(AdminProtect(), Command("pro_add"))
async def pro_add(message: Message):
    args = message.text
    args = args.replace('/pro_add ', '')
    user = await find_user_by_id_or_username(args)
    if not user:
        await message.answer(f"Пользователь {args} не найден в базе.")
        return
    async with async_session() as session:
        await session.execute(update(User).where(User.tg_id == user.tg_id).values(pro=1))
        await session.commit()
    await message.answer(f"PRO подписка выдана пользователю {user.username} ({user.tg_id}).")


@dp.message(AdminProtect(), Command("pro_remove"))
async def pro_remove(message: Message):
    args = message.text
    args = args.replace('/pro_remove ', '')
    user = await find_user_by_id_or_username(args)
    if not user:
        await message.answer(f"Пользователь {args} не найден в базе.")
        return
    async with async_session() as session:
        await session.execute(update(User).where(User.tg_id == user.tg_id).values(pro=0))
        await session.commit()
    await message.answer(f"PRO подписка снята с пользователя {user.username} ({user.tg_id}).")



@dp.message(AdminProtect(), Command('newsletter'))
async def admin(message: Message, state: FSMContext):
    await state.set_state(Newsletter.message)
    await message.answer('Отправь сообщение для рассылки')


@dp.message(AdminProtect(), Newsletter.message)
async def get_admin(message: Message, state: FSMContext, bot: Bot):
    current_state = await state.get_state()
    if current_state is None or current_state != Newsletter.message:
        await message.answer('Сначала начни рассылку, используя команду /newsletter')
        return

    users = await get_users()
    removed_count = 0
    failed_count = 0

    async with async_session() as session:
        for user in users:
            try:
                await bot.send_message(chat_id=user.tg_id, text=message.text)
            except Exception:
                failed_count += 1
                removed_count += 1
                # Удаляем пользователя, который заблокировал бота
                await session.execute(delete(User).where(User.tg_id == user.tg_id))
        await session.commit()

    await message.answer(
        f'Рассылка завершена.\n'
        f'Удалено пользователей: {removed_count}\n'
        f'Не удалось отправить сообщений: {failed_count}'
    )
    await state.clear()


@dp.message(AdminProtect(), Command('users'))
async def how_many(message: Message, bot: Bot):
    async with async_session() as session:
        # Общее количество пользователей
        count_users = await session.scalar(select(func.count(User.id)))
        
        # Количество пользователей с про подпиской
        count_pro_users = await session.scalar(
            select(func.count(User.id)).where(User.pro == 1)
        )
    
    await message.answer(
        f'📊 Статистика пользователей:\n\n'
        f'👥 Всего в базе: {count_users}\n'
        f'💼 С про подпиской: {count_pro_users}\n\n'
    )


@dp.message(AdminProtect(), Command('tokens'))
async def get_tokens(message: Message):
    await message.answer('Токены для работы с RevAI:\n\n' + str(tokens), 
                         reply_markup=InlineKeyboardBuilder().row(InlineKeyboardButton(text='Добавить токен', callback_data='add_token')).row(InlineKeyboardButton(text='Изменить массив полностью', callback_data='edit_tokens')).as_markup())


@dp.message(AdminProtect(), F.text == 'Стоп')
async def stop_adding_tokens(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(f'Токены добавлены. Текущий список: {tokens}', reply_markup=ReplyKeyboardRemove())


@dp.callback_query(AdminProtect(), F.data == 'add_token')
async def add_tokens(callback: CallbackQuery, state: FSMContext):
    await state.set_state(Add_tokens.message)
    await callback.message.answer('Отправляй токены по одному!')


@dp.message(AdminProtect(), Add_tokens.message)
async def add_token(message: Message, state: FSMContext):
    global tokens
    tokens.append(message.text)
    await message.answer('Токен добавлен! Можешь отправить еще', reply_markup=ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text='Стоп')]], resize_keyboard=True))


@dp.message(AdminProtect(), F.text.startswith('[') & F.text.endswith(']'))
async def full_edit_tokens(message: Message):
    global tokens
    tokens = [t.strip().strip("'").strip('"') for t in message.text[1:-1].split(',')]
    await message.answer(f'Токены обновлены. Текущий список: {tokens}')


@dp.message(AdminProtect(), Command('get_db'))
async def get_db(message: Message, bot: Bot):
    database = FSInputFile(
        os.path.abspath(db_path),
        filename='db.sqlite3'
    )
    await message.answer_document(database)
    


async def download_file(session: aiohttp.ClientSession, file_url: str, id_file: str, file_extension: str):
    async with session.get(file_url) as response:
        with open(f'{BASE_DATA_PATH}/{id_file}.{file_extension}', 'wb') as file:
            file.write(await response.read())


def transcribe_file(token: str, id_file: str, duration: int) -> str:
    global tokens
    if duration < 3:
        main_audio = AudioSegment.from_file(f'{BASE_DATA_PATH}/{id_file}.ogg')
        add_audio = AudioSegment.from_file(ogg_path)
        combined = main_audio + add_audio
        combined.export(f'{BASE_DATA_PATH}/{id_file}.ogg', format='ogg')
    client = apiclient.RevAiAPIClient(token)
    job_options = {'language': 'ru'}
    job = client.submit_job_local_file(f'{BASE_DATA_PATH}/{id_file}.ogg', **job_options)
    while True:
        job_details = client.get_job_details(job.id)
        if job_details.status == 'transcribed':
            transcript_text = client.get_transcript_text(job.id)
            os.remove(f'{BASE_DATA_PATH}/{id_file}.ogg')
            transcript_text = re.sub(r'Speaker \d+\s+', '', transcript_text)
            return transcript_text
        if job_details.status == 'failed':
            for tok in tokens:
                if tok == token:
                    tokens.remove(tok)
                    for i in tokens:
                        if len(i) > 2:
                            new_token = i
                    return f'new_{tok}; \n\n tokens={tokens}'


async def download_and_transcribe(bot: Bot, file_id: str, token: str, id_file: str, file_extension: str) -> str:
    file_info = await bot.get_file(file_id)
    file_path = f'https://api.telegram.org/file/bot{API_TOKEN}/{file_info.file_path}'

    async with aiohttp.ClientSession() as session:
        await download_file(session, file_path, id_file, file_extension)

    if file_extension == 'mp4':
        video = VideoFileClip(f'{BASE_DATA_PATH}/{id_file}.mp4')
        video.audio.write_audiofile(f'{BASE_DATA_PATH}/{id_file}.ogg')
        video.close()
        os.remove(f'{BASE_DATA_PATH}/{id_file}.mp4')
    duration = get_duration_pydub(f'{BASE_DATA_PATH}/{id_file}.ogg')
    if await add_usage(id_file[:10], duration):
        await bot.send_message(chat_id='7281169403', text=f'Пользователь {id_file[:11]} превысил лимит времени на аудио')
        return 'Ты превысил лимит времени на аудио, попробуй позже или напиши администратору бота'
    transcript_text = await asyncio.to_thread(transcribe_file, token, id_file, duration)
    if transcript_text[:3] == 'new':
        await bot.send_message(chat_id='7281169403', text=transcript_text)
        transcript_text = 'Извини, сменился токен в приложении(бывает редко, тебе просто не повезло). Отправь заново свое аудио и я его уже точно обработаю)'

    return transcript_text


@dp.message(F.voice)
async def handle_audio_message(message: Message, bot: Bot):
    global tokens
    for i in tokens:
        if len(i) > 2:
            token = i
    await message.answer('Уже обрабатываю, подожди немного...')
    await create_user(message.from_user.id, str(message.from_user.username))
    num_file = str(message.from_user.id) + str(random.randint(1, 999999))
    voice_file_id = message.voice.file_id
    try:
        transcript_text = await download_and_transcribe(bot, voice_file_id, token, str(num_file), 'ogg')
        if len(transcript_text)<4096:
            await message.reply(transcript_text)
        else:
            for i in range(int(len(transcript_text)//4096)+1):
                await message.answer(transcript_text[i*4096:(i+1)*4096])
    except TelegramBadRequest:
        await message.reply('Текст в этом аудио отсутствует')


@dp.message(F.video_note)
async def handle_video_message(message: Message):
    global tokens
    for i in tokens:
        if len(i) > 2:
            token = i
    await message.answer('Уже обрабатываю, подожди немного...')
    video_file_id = message.video_note.file_id
    num_file = str(message.from_user.id) + str(random.randint(1, 999999))
    try:
        transcript_text = await download_and_transcribe(bot, video_file_id, token, num_file, 'mp4')
        await message.reply(transcript_text)
    except TelegramBadRequest:
        await message.reply('Текст в этом аудио отсутствует')


async def main():
    await async_main()
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
