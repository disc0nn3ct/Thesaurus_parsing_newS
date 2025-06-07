import os
import subprocess
import sys
import logging

logger = logging.getLogger(__name__)  


# def check_git_update(commit_file="log/current_commit.txt"):
def check_git_update(commit_file="current_commit.txt"):
    
    try:
        # Переход в директорию, где находится запускаемый скрипт
        project_root = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.chdir(project_root)
        logger.info(f"📁 Перешли в директорию проекта: {project_root}")

        os.makedirs(os.path.dirname(commit_file), exist_ok=True)
        subprocess.run(["git", "fetch"], check=True)
        new_commit = subprocess.check_output(
            ["git", "rev-parse", "origin/main"], text=True
        ).strip()

        if not os.path.exists(commit_file):
            with open(commit_file, "w") as f:
                f.write(new_commit)
            logger.info(f"📄 Создан файл {commit_file}, установлен текущий коммит: {new_commit}")
            return None

        with open(commit_file, "r") as f:
            last_commit = f.read().strip()

        if new_commit != last_commit:
            logger.info(f"🔄 Обнаружено обновление: {new_commit}")
            return new_commit
        else:
            logger.info("✅ Локальная версия актуальна.")
            return None

    except Exception as e:
        logger.exception("❌ Ошибка при проверке обновления Git:")
        return None


# # def update_and_restart(new_commit, commit_file="log/current_commit.txt"):
# def update_and_restart(new_commit, commit_file="current_commit.txt"):

#     try:
#         # Убедиться, что работаем из директории скрипта
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         os.chdir(script_dir)
        
#         subprocess.run(["git", "pull", "origin", "main"], check=True)
#         with open(commit_file, "w") as f:
#             f.write(new_commit)

#         logger.info("♻️ Проект обновлён. Перезапуск...")
#         os.execv(sys.executable, [sys.executable] + sys.argv)

#     except Exception as e:
#         logger.exception("❌ Ошибка при обновлении и перезапуске:")

def update_and_restart(new_commit, commit_file):
    try:
        subprocess.run(["git", "pull", "origin", "main"], check=True)

        with open(commit_file, "w") as f:
            f.write(new_commit)

        logger.info("♻️ Проект обновлён. Перезапуск...")
        os.execv(sys.executable, [sys.executable] + sys.argv)

    except Exception as e:
        logger.exception("❌ Ошибка при обновлении и перезапуске:")





def check_and_restart_if_updated(commit_file="log/current_commit.txt"):
# def check_and_restart_if_updated(commit_file="log/current_commit.txt"):
    new_commit = check_git_update(commit_file)
    if new_commit:
        update_and_restart(new_commit, commit_file)



# def check_and_restart_if_updated(commit_file="log/current_commit.txt"):
#     project_root = os.path.dirname(os.path.abspath(sys.argv[0]))
#     full_commit_path = os.path.join(project_root, commit_file)
#     os.makedirs(os.path.dirname(full_commit_path), exist_ok=True)

#     new_commit = check_git_update(full_commit_path)
#     if new_commit:
#         update_and_restart(new_commit, full_commit_path)


# def check_git_update(commit_file):
#     try:
#         subprocess.run(["git", "fetch"], check=True)
#         new_commit = subprocess.check_output(["git", "rev-parse", "origin/main"], text=True).strip()

#         if not os.path.exists(commit_file):
#             with open(commit_file, "w") as f:
#                 f.write(new_commit)
#             logger.info(f"📄 Создан файл {commit_file}, установлен текущий коммит: {new_commit}")
#             return None

#         with open(commit_file, "r") as f:
#             last_commit = f.read().strip()

#         if new_commit != last_commit:
#             logger.info(f"🔄 Обнаружено обновление: {new_commit}")
#             return new_commit
#         else:
#             logger.info("✅ Локальная версия актуальна.")
#             return None

#     except Exception as e:
#         logger.exception("❌ Ошибка при проверке обновления Git:")
#         return None

# def update_and_restart(new_commit, commit_file):
#     try:
#         subprocess.run(["git", "pull", "origin", "main"], check=True)

#         with open(commit_file, "w") as f:
#             f.write(new_commit)

#         logger.info("♻️ Проект обновлён. Перезапуск...")
#         os.execv(sys.executable, [sys.executable] + sys.argv)

#     except Exception as e:
#         logger.exception("❌ Ошибка при обновлении и перезапуске:")
