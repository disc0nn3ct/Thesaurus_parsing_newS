# Thesaurus

import pandas as pd
from datetime import datetime, timedelta, time
import os
import requests
from requests.exceptions import RequestException
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import logging


import holidays
from datetime import date

# Настройка логгера
log_dir = os.path.join(os.getcwd(), "log")
os.makedirs(log_dir, exist_ok=True)

# Полный путь к лог-файлу
log_file = os.path.join(log_dir, "ruonia_log.txt")

# Настройка логгера
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)

logger = logging.getLogger(__name__)


def is_russian_workday(check_date=None):
    if check_date is None:
        check_date = date.today()

    ru_holidays = holidays.Russia()
    return check_date.weekday() < 5 and check_date not in ru_holidays


# Проверяет за какой промежутек нужен запрос данных и загружает актуальную информацию на сегодня, 
def check_if_need_new_rec(FILENAME="ruonia_data.xlsx"):
    try:
        today = datetime.today().date()
        end_date = today.strftime("%d.%m.%Y")
        start_date = "11.01.2010"
        from_dt = datetime.strptime(start_date, "%d.%m.%Y").strftime("%m/%d/%Y")
        to_dt = today.strftime("%m/%d/%Y")

        if os.path.exists(FILENAME):
            try:
                local_df = pd.read_excel(FILENAME)
                local_df["Дата"] = pd.to_datetime(local_df["Дата"], dayfirst=True)
                last_date = local_df["Дата"].max().date()
                from_date = local_df["Дата"].min().date()

                logger.info(f"Файл найден. С {from_date} по {last_date}")

                if not is_russian_workday():
                    logger.info("Сегодня выходной или праздник — обновление не требуется.")
                    return 0

                if last_date.strftime('%d.%m.%Y') == today.strftime('%d.%m.%Y') or (
                    last_date.strftime('%d.%m.%Y') == (today - timedelta(days=1)).strftime('%d.%m.%Y') and
                    datetime.now().time() < time(14, 0)
                ):
                    logger.info("Данные уже актуальны. Обновление не требуется.")
                    return 0
                else:
                    logger.info("Обнаружены новые данные. Загружаем обновление.")
            except Exception as e:
                logger.warning(f"Ошибка чтения файла: {e}")
                logger.info("Выполняем загрузку заново.")
        else:
            logger.info(f"Файл {FILENAME} не найден. Загружаем с {start_date} по {end_date}.")

        url = f"https://cbr.ru/Queries/UniDbQuery/DownloadExcel/125022?Posted=True&From={start_date}&To={end_date}&I1=true&M1=true&M3=true&M6=true&FromDate={from_dt}&ToDate={to_dt}"
        logger.info(f"Запрос по ссылке: {url}")

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса данных: {e}")
            return -1

        with open(FILENAME, "wb") as f:
            f.write(response.content)
            logger.info(f"Файл успешно сохранён: {FILENAME}")

        return 1

    except RequestException as e:
        logger.error(f"Ошибка загрузки с сайта ЦБ: {e}")
        return -1
    except Exception as e:
        logger.exception(f"Необработанная ошибка: {e}")
        return -2



# Построение графиков, похоже на скользящие средние 
def analitics(FILENAME="ruonia_data.xlsx"):
    # Создаем имя файла PNG с текущей датой
    today_str = datetime.today().strftime("%Y-%m-%d")
    base_filename = f"ruonia_trend_{today_str}"
    ext = ".png"

    # Папка для сохранения
    output_dir = os.path.join(os.getcwd(), "src")
    os.makedirs(output_dir, exist_ok=True)  # создаём, если нет

    # Проверка существующих файлов и добавление версии
    version = 1
    output_path = os.path.join(output_dir, base_filename + ext)
    while os.path.exists(output_path):
        version += 1
        output_path = os.path.join(output_dir, f"{base_filename}_v{version}{ext}")

    try:
        # Загружаем и обрабатываем данные
        df = pd.read_excel(FILENAME)
        df = df.rename(columns={
            "Индекс": "RUONIA",
            "1 месяц": "1 мес",
            "3 месяца": "3 мес",
            "6 месяцев": "6 мес"
        })

        df["Дата"] = pd.to_datetime(df["Дата"], dayfirst=True)
        df = df.dropna(subset=["RUONIA", "1 мес", "3 мес", "6 мес"])
        df = df.sort_values("Дата")

        # Строим график
        plt.figure(figsize=(14, 7))
        plt.plot(df["Дата"], df["RUONIA"], label="RUONIA (overnight)", linewidth=2)
        plt.plot(df["Дата"], df["1 мес"], label="RUONIA 1 мес", linestyle="--")
        plt.plot(df["Дата"], df["3 мес"], label="RUONIA 3 мес", linestyle="-.")
        plt.plot(df["Дата"], df["6 мес"], label="RUONIA 6 мес", linestyle=":")

        plt.title("Динамика индекса RUONIA и срочных ставок", fontsize=14)
        plt.xlabel("Дата")
        plt.ylabel("Ставка (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Сохраняем график
        plt.savefig(output_path)
        plt.close()

        logger.info(f"📈 График успешно сохранён: {output_path}")
        return output_path

    except Exception as e:
        logger.exception(f"❌ Ошибка при построении графика: {e}")
        return None
    
#проветси анализ РУОНИИ 
def make_analyze_ruonia(filepath="ruonia_data.xlsx"):
    try:
        df = pd.read_excel(filepath)
        df = df.rename(columns={
            "Индекс": "RUONIA",
            "1 месяц": "1 мес",
            "3 месяца": "3 мес",
            "6 месяцев": "6 мес"
        })
        df["Дата"] = pd.to_datetime(df["Дата"], dayfirst=True)
        df = df.sort_values("Дата")

        last_30 = df.tail(30)
        last_15 = last_30.tail(15)
        last_10 = last_30.tail(10)

        latest_date = last_10["Дата"].iloc[-1].strftime("%d.%m.%Y")
        previous_date = last_10["Дата"].iloc[-2].strftime("%d.%m.%Y")

        indicators = ["RUONIA", "1 мес", "3 мес", "6 мес"]
        full_text = f"📅 Последняя дата данных: {latest_date}\n"

        for col in indicators:
            latest = last_10[col].iloc[-1]
            previous = last_10[col].iloc[-2]

            delta_1 = latest - previous
            delta_10 = last_10[col].iloc[-1] - last_10[col].iloc[0]
            delta_15 = last_15[col].iloc[-1] - last_15[col].iloc[0]
            delta_30 = last_30[col].iloc[-1] - last_30[col].iloc[0]

            mean_10 = last_10[col].mean()
            mean_15 = last_15[col].mean()
            mean_30 = last_30[col].mean()

            if delta_10 > 0 and delta_15 > 0 and delta_30 > 0:
                trend = "📈 плавный восходящий тренд"
            elif delta_10 < 0 and delta_15 < 0 and delta_30 < 0:
                trend = "📉 стабильное снижение"
            else:
                trend = "📊 неопределённое поведение"

            full_text += (
                f"\n📌 **{col}**\n"
                f"• Сегодня: {latest:.4f}\n"
                f"• Вчера ({previous_date}): {previous:.4f}\n"
                f"• Δ за день: {delta_1:+.4f}\n"
                f"• Среднее за 10 дней: {mean_10:.4f}\n"
                f"• Среднее за 15 дней: {mean_15:.4f}\n"
                f"• Среднее за 30 дней: {mean_30:.4f}\n"
                f"• Рост за 10 дней: {delta_10:+.4f}\n"
                f"• Рост за 15 дней: {delta_15:+.4f}\n"
                f"• Рост за 30 дней: {delta_30:+.4f}\n"
                f"• Тренд: {trend}\n"
            )

        logger.info("🧾 Анализ RUONIA успешно выполнен.")
        logger.debug(f"\n{full_text}")
        return full_text

    except Exception as e:
        logger.exception(f"❌ Ошибка в аналитике RUONIA: {e}")
        return None


def send_info_ruonia(client, recipients):
    folder_path = os.path.join(os.getcwd(), "src")
    base_name = "ruonia_trend_"
    extension = ".png"

    # Получаем список подходящих файлов
    matching_files = [
        f for f in os.listdir(folder_path)
        if f.startswith(base_name) and f.endswith(extension)
    ] if os.path.exists(folder_path) else []

    if matching_files:
        matching_files.sort(reverse=True)
        latest_file = os.path.join(folder_path, matching_files[0])
        logger.info(f"📂 Найден последний график: {latest_file}")
    else:
        logger.warning("📂 График не найден. Генерируем с помощью analitics()...")
        latest_file = analitics()

    if not latest_file or not os.path.exists(latest_file):
        logger.error("❌ Не удалось найти или создать файл графика RUONIA.")
        return
    analysis = make_analyze_ruonia()

    for chat_id in recipients:
        try:
            logger.info(f"📤 Отправка графика и анализа в чат: {chat_id}")
            client.send_photo(
                chat_id,
                photo=latest_file,
                caption="📈 График RUONIA за всё время"
            )
            if analysis:
                client.send_message(chat_id, analysis)
                logger.info(f"✅ Анализ успешно отправлен в {chat_id}")
            else:
                logger.warning(f"⚠️ Анализ не сгенерирован — сообщение не отправлено в {chat_id}")
        except Exception as e:
            logger.exception(f"⚠️ Ошибка при отправке в {chat_id}: {e}")

# https://cbr.ru/Queries/UniDbQuery/DownloadExcel/125022?Posted=True&From=11.01.2010&To=30.04.2025&I1=true&M1=true&M3=true&M6=true&FromDate=01%2F11%2F2010&ToDate=04%2F30%2F2025


#################################### Вернуть ######
# check_if_need_new_rec()

# analitics()  # Либо переделать 
#################################### Вернуть ######


###### Проверка обновления ####
import subprocess
import os
import sys

def check_git_update(commit_file="log/current_commit.txt"):
    try:
        # Убедимся, что папка log существует
        os.makedirs(os.path.dirname(commit_file), exist_ok=True)

        # Получаем текущий коммит с origin
        subprocess.run(["git", "fetch"], check=True)
        new_commit = subprocess.check_output(
            ["git", "rev-parse", "origin/main"], text=True
        ).strip()

        # Если файла нет — создаём и записываем текущий коммит
        if not os.path.exists(commit_file):
            with open(commit_file, "w") as f:
                f.write(new_commit)
            logger.info(f"📄 Файл {commit_file} создан. Установлен коммит: {new_commit}")
            return None  # Первый запуск — обновление не требуется

        # Считываем сохранённый коммит
        with open(commit_file, "r") as f:
            last_commit = f.read().strip()

        if new_commit != last_commit:
            logger.info(f"🔄 Обнаружен новый коммит: {new_commit}")
            return new_commit
        else:
            logger.info("✅ Версия актуальна. Обновление не требуется.")
            return None

    except Exception as e:
        logger.exception("❌ Ошибка при проверке обновления Git:")
        return None


def update_and_restart(new_commit, commit_file="log/current_commit.txt"):
    try:
        subprocess.run(["git", "pull"], check=True)

        with open(commit_file, "w") as f:
            f.write(new_commit)

        logger.info("♻️ Проект обновлён. Перезапускаем...")
        os.execv(sys.executable, ['python'] + sys.argv)

    except Exception as e:
        logger.exception("❌ Ошибка при обновлении и перезапуске:")



commit_file = "log/current_commit.txt"
new_commit = check_git_update(commit_file)
if new_commit:
    update_and_restart(new_commit, commit_file)


######

load_dotenv()  

api_hash = os.getenv('api_hash')
for_whom = os.getenv('for_whom')
api_id = os.getenv('api_id')
bot_token = os.getenv('bot_token')

recipients_raw = os.getenv("for_whom_list", "")
recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
if not recipients:
    raise ValueError("❌ Нет получателей. Убедись, что for_whom_list задан в .env")




from pyrogram import Client, idle

# #################  вернуть 
# client = Client(name='me_client', api_id=api_id, api_hash=api_hash, bot_token = bot_token )
# # Запуск клиента
# client.start()

        


# check_if_need_new_rec()
# send_info_ruonia(client, recipients)



# # idle()

# # Завершение сессии
# client.stop()

# #################### вернуть 







