# Thesaurus

import pandas as pd
from datetime import datetime, timedelta, time
import os
import requests
from requests.exceptions import RequestException
import matplotlib.pyplot as plt
from dotenv import load_dotenv


import holidays
from datetime import date

def is_russian_workday(check_date=None):
    if check_date is None:
        check_date = date.today()

    ru_holidays = holidays.Russia()
    return check_date.weekday() < 5 and check_date not in ru_holidays


# Проверяет за какой промежутек нужен запрос данных и загружает актуальную информацию на сегодня, 
def check_if_need_new_rec(FILENAME="ruonia_data.xlsx"):
    try:
        # Сегодняшняя дата
        today = datetime.today().date()
        end_date = today.strftime("%d.%m.%Y")
        start_date = "11.01.2010"

        from_dt = datetime.strptime(start_date, "%d.%m.%Y").strftime("%m/%d/%Y")
        to_dt = today.strftime("%m/%d/%Y")

        # Проверка существующего файла
        if os.path.exists(FILENAME):
            try:
                local_df = pd.read_excel(FILENAME)
                local_df["Дата"] = pd.to_datetime(local_df["Дата"], dayfirst=True)
                last_date = local_df["Дата"].max().date()
                from_date = local_df["Дата"].min().date()
                print(f"Файл начинается с {from_date.strftime('%d.%m.%Y')}, а последняя дата в файле: {last_date.strftime('%d.%m.%Y')}")
                # print("================== ", last_date.strftime('%d.%m.%Y'), " ================= ", end_date, " ======= ", today.strftime('%d.%m.%Y'))
                if not is_russian_workday():
                        print("Сегодня выходной или праздник в РФ — обновление RUONIA не требуется.")
                        return 0 # так как это выходной
                if last_date.strftime('%d.%m.%Y') == today.strftime('%d.%m.%Y') or ( last_date.strftime('%d.%m.%Y') == ((today-timedelta(days=1)).strftime('%d.%m.%Y')) and datetime.now().time() < time(14, 0)): # И дата - 1 если часов меньше 14 
                    print("Данные уже актуальны. Обновление не требуется.")
                    return 0
                else:
                    print("Появилась новая дата. Загружаем обновлённый файл.")
            except Exception as e:
                print(f"⚠️ Ошибка при чтении существующего файла: {e}")
                print("Будет выполнена загрузка заново.")
        else:
            print(f"Файл {FILENAME} не найден. Загружаем с {start_date} по {end_date}.")
        
        # Формируем URL
        url = f"https://cbr.ru/Queries/UniDbQuery/DownloadExcel/125022?Posted=True&From={start_date}&To={end_date}&I1=true&M1=true&M3=true&M6=true&FromDate={from_dt}&ToDate={to_dt}"



        # Скачиваем файл
        print(f"Скачиваем данные с ЦБ РФ по ссылке: {url}")
        response = requests.get(url)

        try:
            response = requests.get(url, timeout=10)  # таймаут на случай зависания
            response.raise_for_status()  # выбросит ошибку, если код ответа ≠ 200
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка при запросе файла с сайта ЦБ: {e}")
            return -1
    
        with open(FILENAME, "wb") as f:
            f.write(response.content)
            print(f"Файл сохранён как: {FILENAME}")
            f.close()
        return 1

    except RequestException as e:
        print(f"❌ Ошибка при загрузке данных с сайта ЦБ: {e}")
        return -1
    except Exception as e:
        print(f"❌ Необработанная ошибка: {e}")
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

        print(f"✅ График сохранён: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Ошибка при построении графика: {e}")
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

            # Определение тренда
            if delta_10 > 0 and delta_15 > 0 and delta_30 > 0:
                trend = "📈 плавный восходящий тренд"
            elif delta_10 < 0 and delta_15 < 0 and delta_30 < 0:
                trend = "📉 стабильное снижение"
            else:
                trend = "📊 неопределённое поведение"

            # Формируем блок отчёта по индикатору
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

        print(full_text)
        return full_text

    except Exception as e:
        print(f"❌ Ошибка в аналитике RUONIA: {e}")
        return None

def send_info_ruonia(client, recipients):

    folder_path = os.path.join(os.getcwd(), "src")
    base_name = "ruonia_trend_"
    extension = ".png"
    # Находим все версии файлов с нужным шаблоном
    matching_files = [
        f for f in os.listdir(folder_path)
        if f.startswith(base_name) and f.endswith(extension)
    ]

    # Если есть файлы, выбираем тот, у которого версия (или дата) максимальна
    if matching_files:
        # Сортируем по убыванию, предполагая, что последние версии идут позже
        matching_files.sort(reverse=True)
        latest_file = os.path.join(folder_path, matching_files[0])
    else:
        latest_file = None

    for chat_id in recipients:
        print("========================= ", chat_id )
        client.send_photo(
            chat_id,
            photo=latest_file,
            caption="📈 График RUONIA за всё время" + latest_file
        )
        client.send_message(chat_id, make_analyze_ruonia())









# https://cbr.ru/Queries/UniDbQuery/DownloadExcel/125022?Posted=True&From=11.01.2010&To=30.04.2025&I1=true&M1=true&M3=true&M6=true&FromDate=01%2F11%2F2010&ToDate=04%2F30%2F2025


#################################### Вернуть ######
check_if_need_new_rec()

# analitics()
#################################### Вернуть ######


load_dotenv()  

api_hash = os.getenv('api_hash')
for_whom = os.getenv('for_whom')
api_id = os.getenv('api_id')
bot_token = os.getenv('bot_token')

recipients_raw = os.getenv("for_whom_list", "")




from pyrogram import Client


client = Client(name='me_client', api_id=api_id, api_hash=api_hash, bot_token = bot_token )
# Запуск клиента
client.start()


recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]

if not recipients:
    raise ValueError("❌ Нет получателей. Убедись, что for_whom_list задан в .env")
        
send_info_ruonia(client, recipients)






# Завершение сессии
client.stop()


#### Либо так ### 
# with Client(name='me_client', api_id=api_id, api_hash=api_hash, bot_token=bot_token) as client:
#     client.send_message(for_whom, "📊 Привет! Это автоматическое сообщение от Pyrogram-бота.")


####

# client.send_message(for_whom, "📊 Привет! Это автоматическое сообщение от Pyrogram-бота.")






