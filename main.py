# Thesaurus

import pandas as pd
from datetime import datetime, timedelta, time
import os
import requests
from requests.exceptions import RequestException
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import logging
import matplotlib.dates as mdates
import re

import sys


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




    return check_date not in ru_holidays # выходные сб вс и    
    # return check_date.weekday() < 5 and check_date not in ru_holidays # выходные сб вс и


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
                    # logger.info("Сегодня выходной или праздник — обновление не требуется.")
                    logger.info("Праздник — обновление не требуется.")
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
    today_str = datetime.today().strftime("%Y-%m-%d")
    base_filename = f"ruonia_trend_{today_str}"
    ext = ".png"

    output_dir = os.path.join(os.getcwd(), "src")
    os.makedirs(output_dir, exist_ok=True)

    # Генерация имени для полного графика
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

        # --- 📈 График со всеми данными ---
        plt.figure(figsize=(14, 7))
        plt.plot(df["Дата"], df["RUONIA"], label="RUONIA (overnight)", linewidth=2)
        plt.plot(df["Дата"], df["1 мес"], label="RUONIA 1 мес", linestyle="--")
        plt.plot(df["Дата"], df["3 мес"], label="RUONIA 3 мес", linestyle="-.")
        plt.plot(df["Дата"], df["6 мес"], label="RUONIA 6 мес", linestyle=":")

        plt.title(f"Динамика индекса RUONIA и срочных ставок до {today_str}", fontsize=14)
        plt.xlabel("Дата")
        plt.ylabel("Ставка (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        logger.info(f"📈 График (все данные) сохранён: {output_path}")

        # --- 📉 График за последние 90 дней ---
        short_df = df[df["Дата"] >= (datetime.today() - timedelta(days=90))]

        plt.figure(figsize=(14, 7))
        plt.plot(short_df["Дата"], short_df["1 мес"], label="RUONIA 1 мес", linestyle="--")
        plt.plot(short_df["Дата"], short_df["3 мес"], label="RUONIA 3 мес", linestyle="-.")
        plt.plot(short_df["Дата"], short_df["6 мес"], label="RUONIA 6 мес", linestyle=":")

        plt.title(f"RUONIA (последние 90 дней) до {today_str}", fontsize=14)
        plt.xlabel("Дата")
        plt.ylabel("Ставка (%)")
        plt.legend()
        plt.grid(True)

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        plt.xticks(rotation=90)

        plt.tight_layout()

        # Сохраняем второй файл с _last90
        short_filename = f"{base_filename}_last90"
        short_output_path = os.path.join(output_dir, short_filename + ext)
        version = 1
        while os.path.exists(short_output_path):
            version += 1
            short_output_path = os.path.join(output_dir, f"{short_filename}_v{version}{ext}")

        plt.savefig(short_output_path)
        plt.close()

        logger.info(f"📉 График (последние 90 дней) сохранён: {short_output_path}")
        return output_path, short_output_path

    except Exception as e:
        logger.exception(f"❌ Ошибка при построении графиков: {e}")
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
    short_base_name = "ruonia_trend_"
    short_suffix = "_last90"
    extension = ".png"

    # Получаем список подходящих файлов
    matching_files = [
        f for f in os.listdir(folder_path)
        if f.startswith(base_name) and f.endswith(extension) and short_suffix not in f
    ] if os.path.exists(folder_path) else []

    # Получаем список коротких графиков (last90)
    matching_short_files = [
        f for f in os.listdir(folder_path)
        if f.startswith(short_base_name) and short_suffix in f and f.endswith(extension)
    ] if os.path.exists(folder_path) else []

    # if matching_files:
    #     matching_files.sort(reverse=True)
    #     latest_file = os.path.join(folder_path, matching_files[0])
    #     logger.info(f"📂 Найден последний график: {latest_file}")
    # else:
    #     logger.warning("📂 График не найден. Генерируем с помощью analitics()...")
    #     latest_file = analitics()

    #Заменил на всегда генерацию
    logger.warning("📂 Всегда!!!!. Генерируем с помощью analitics()...")
    # latest_file, latest_short_file = analitics()

    # latest_file = analitics()
    result = analitics()
    if not result:
        logger.error("❌ Не удалось создать графики.")

    latest_file, latest_short_file = result

    if not os.path.exists(latest_file):
        logger.error("❌ Файл графика не найден после генерации.")


    # Поиск соответствующего short-файла
    latest_short_file = None
    if matching_short_files:
        matching_short_files.sort(reverse=True)
        latest_short_file = os.path.join(folder_path, matching_short_files[0])
        logger.info(f"📂 Найден короткий график (90 дней): {latest_short_file}")





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
                caption="📈 График RUONIA за всё время до " + datetime.today().strftime("%Y-%m-%d")
            )

            # Отправка дополнительного графика (last90), если найден
            if latest_short_file and os.path.exists(latest_short_file):
                client.send_photo(
                    chat_id,
                    photo=latest_short_file,
                    caption="📉 RUONIA за последние 90 дней"
                )

            if analysis:
                client.send_message(chat_id, analysis)
                logger.info(f"✅ Анализ успешно отправлен в {chat_id}")
            else:
                logger.warning(f"⚠️ Анализ не сгенерирован — сообщение не отправлено в {chat_id}")
        except Exception as e:
            logger.exception(f"⚠️ Ошибка при отправке в {chat_id}: {e}")


# https://cbr.ru/Queries/UniDbQuery/DownloadExcel/125022?Posted=True&From=11.01.2010&To=30.04.2025&I1=true&M1=true&M3=true&M6=true&FromDate=01%2F11%2F2010&ToDate=04%2F30%2F2025

###########################################################################################################################AI
# from functions.ai import run_brief

# # ---------- основная функция с контекст-менеджером ----------
# def send_ai(client, recipients):
#     """
#     Получает ответ от модели (Markdown) через functions.ai.run_brief и рассылает его:
#       • СНАЧАЛА сообщениями (по две главы в одном сообщении, укладываясь в лимит Telegram),
#       • затем прикладывает один архивный .md файл в папку src/ai/ и отправляет в чат.

#     Устойчива к разным сигнатурам run_brief():
#       - может вернуть (answer),
#       - или (answer, tokens),
#       - или (answer, tokens, *anything_else).  
#     В любом случае нормализует Markdown (закрывает ```), делит на главы `## N. ...`.
#     При ошибке разметки повторяет отправку без parse_mode.
#     """
#     import os
#     import re
#     from datetime import datetime
#     from functions.ai import run_brief

#     TELEGRAM_LIMIT = 4000

#     # --- helpers -----------------------------------------------------------
#     def normalize_code_fences(text: str) -> str:
#         # Приводим ```md/markdown к простому ``` и закрываем незакрытые
#         text = re.sub(r"```\s*(markdown|md|Markdown)\s*\n", "```\n", text)
#         if text.count("```") % 2 == 1:
#             text = text.rstrip() + "\n\n```\n"
#         return text

#     def split_into_chapters(md: str):
#         # Глава = заголовок вида: ## 1. ...
#         header_pat = re.compile(r"^##\s+\d+\.[\t ]*.*$", re.M)
#         matches = list(header_pat.finditer(md))
#         if not matches:
#             return [md]
#         parts = []
#         first_start = matches[0].start()
#         prologue = md[:first_start].strip("\n")
#         if prologue:
#             parts.append(prologue)
#         for i, m in enumerate(matches):
#             start = m.start()
#             end = matches[i+1].start() if i+1 < len(matches) else len(md)
#             parts.append(md[start:end].strip("\n"))
#         return parts

#     def split_hard(block: str, limit: int):
#         # Абзацы -> строки -> символы, нормализуя ``` в каждом фрагменте
#         parts, cur = [], ""
#         def flush():
#             nonlocal cur
#             if cur.strip():
#                 parts.append(normalize_code_fences(cur).strip())
#                 cur = ""
#         for p in block.split("\n\n"):
#             chunk = p + "\n\n"
#             if len(chunk) > limit:
#                 for ln in chunk.splitlines(True):
#                     if len(ln) > limit:
#                         for s in range(0, len(ln), limit):
#                             part = ln[s:s+limit]
#                             if cur and len(cur)+len(part) > limit:
#                                 flush()
#                             cur += part
#                     else:
#                         if cur and len(cur)+len(ln) > limit:
#                             flush()
#                         cur += ln
#             else:
#                 if cur and len(cur)+len(chunk) > limit:
#                     flush()
#                 cur += chunk
#         flush()
#         return parts

#     def bundle_messages(chapters, limit: int):
#         # Склеиваем по 2 главы, уважая лимит. Длинные главы режем.
#         msgs = []
#         i, n = 0, len(chapters)
#         while i < n:
#             a = normalize_code_fences(chapters[i])
#             if i + 1 < n:
#                 b = normalize_code_fences(chapters[i+1])
#                 if len(a) + len(b) <= limit:
#                     msgs.append((a + "\n\n" + b).strip())
#                     i += 2
#                     continue
#             if len(a) > limit:
#                 msgs.extend(split_hard(a, limit))
#             else:
#                 msgs.append(a)
#             i += 1
#         return msgs

#     # --- get model answer --------------------------------------------------
#     try:
#         result = run_brief()
#     except Exception as e:
#         for chat_id in recipients:
#             try:
#                 client.send_message(chat_id, f"❌ Ошибка генерации AI-brief: {e}")
#             except Exception:
#                 pass
#         return

#     # Нормализуем возвращаемое значение под (answer, tokens)
#     answer, tokens = None, None
#     if isinstance(result, tuple):
#         if len(result) >= 1:
#             answer = result[0]
#         if len(result) >= 2:
#             tokens = result[1]
#     else:
#         answer = result

#     if not isinstance(answer, str) or not answer.strip():
#         for chat_id in recipients:
#             try:
#                 client.send_message(chat_id, "⚠️ Пустой ответ от модели.")
#             except Exception:
#                 pass
#         return

#     # --- prepare text & files ---------------------------------------------
#     answer = normalize_code_fences(answer)

#     # Сохраняем один .md файл для архива и возможного fallback
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     base_dir = os.path.join(os.getcwd(), "src", "ai")
#     os.makedirs(base_dir, exist_ok=True)
#     md_path = os.path.join(base_dir, f"ai_brief_{ts}.md")
#     with open(md_path, "w", encoding="utf-8") as f:
#         f.write(answer)
#         if tokens:
#             f.write("\n\n" + str(tokens))

#     # Формируем список сообщений (одним, если умещается)
#     if len(answer) <= TELEGRAM_LIMIT:
#         messages = [answer]
#     else:
#         chapters = split_into_chapters(answer) or [answer]
#         messages = bundle_messages(chapters, TELEGRAM_LIMIT)

#     # --- send: messages first, then file ----------------------------------
#     for chat_id in recipients:
#         # 1) Сначала пробуем Markdown
#         sent_msgs = True
#         try:
#             for i, msg in enumerate(messages, 1):
#                 prefix = f"Часть {i}/{len(messages)}\n\n" if len(messages) > 1 else ""
#                 client.send_message(chat_id, prefix + msg, parse_mode="Markdown")
#         except Exception:
#             sent_msgs = False

#         # 2) Если не получилось — отправим без parse_mode
#         if not sent_msgs:
#             try:
#                 for i, msg in enumerate(messages, 1):
#                     prefix = f"Часть {i}/{len(messages)}\n\n" if len(messages) > 1 else ""
#                     client.send_message(chat_id, prefix + msg)
#                 sent_msgs = True
#             except Exception:
#                 sent_msgs = False

#         # 3) Короткое сообщение про токены
#         if tokens and sent_msgs:
#             try:
#                 client.send_message(chat_id, str(tokens))
#             except Exception:
#                 pass

#         # 4) И один архивный .md файл
#         try:
#             client.send_document(chat_id, md_path, caption="📄 AI-brief (.md)")
#         except Exception:
#             pass



###########################################################################################################################AI


#################################### Вернуть ######
# check_if_need_new_rec()

# analitics()  # Либо переделать 
#################################### Вернуть ######


###### Проверка обновления ####
# import subprocess
# import os
# import sys

# def check_git_update(commit_file="log/current_commit.txt"):
#     try:
#         # Убедимся, что папка log существует
#         os.makedirs(os.path.dirname(commit_file), exist_ok=True)

#         # Получаем текущий коммит с origin
#         subprocess.run(["git", "fetch"], check=True)
#         new_commit = subprocess.check_output(
#             ["git", "rev-parse", "origin/main"], text=True
#         ).strip()

#         # Если файла нет — создаём и записываем текущий коммит
#         if not os.path.exists(commit_file):
#             with open(commit_file, "w") as f:
#                 f.write(new_commit)
#             logger.info(f"📄 Файл {commit_file} создан. Установлен коммит: {new_commit}")
#             return None  # Первый запуск — обновление не требуется

#         # Считываем сохранённый коммит
#         with open(commit_file, "r") as f:
#             last_commit = f.read().strip()

#         if new_commit != last_commit:
#             logger.info(f"🔄 Обнаружен новый коммит: {new_commit}")
#             return new_commit
#         else:
#             logger.info("✅ Версия актуальна. Обновление не требуется.")
#             return None

#     except Exception as e:
#         logger.exception("❌ Ошибка при проверке обновления Git:")
#         return None


# def update_and_restart(new_commit, commit_file="log/current_commit.txt"):
#     try:
#         subprocess.run(["git", "pull"], check=True)

#         with open(commit_file, "w") as f:
#             f.write(new_commit)

#         logger.info("♻️ Проект обновлён. Перезапускаем...")
#         os.execv(sys.executable, ['python'] + sys.argv)

#     except Exception as e:
#         logger.exception("❌ Ошибка при обновлении и перезапуске:")



# commit_file = "log/current_commit.txt"
# new_commit = check_git_update(commit_file)
# if new_commit:
#     update_and_restart(new_commit, commit_file)



################################################################################ВЕРНУТЬ
from functions.auto_update import check_and_restart_if_updated
check_and_restart_if_updated()
#################################################################################ВЕРНУТЬ
# ######

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

#################  вернуть 
client = Client(name='me_client', api_id=api_id, api_hash=api_hash, bot_token = bot_token )
# Запуск клиента
client.start()

        


check_if_need_new_rec()
# send_info_ruonia(client, recipients)
send_ai(client, recipients)


# idle()

# Завершение сессии
client.stop()

#################### вернуть 







