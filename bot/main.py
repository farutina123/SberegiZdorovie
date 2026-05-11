from __future__ import annotations

import logging
import os

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from bot.config import load_config
from bot.context import load_context
from bot.llm import ask_llm
from bot.memory import ChatMemory
from bot.structured_answers import extract_eeg_facts
from bot.system_prompt import SYSTEM_PROMPT


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("sberegi-bot")

def _ensure_ca_bundle() -> None:
    """
    Force a known CA bundle (certifi) for TLS verification.
    Helps on Windows/corporate networks where Python can't find system roots.
    """
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE"):
        return
    try:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
    except Exception:
        # If certifi isn't available, keep default behavior.
        return


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Внутренний ассистент кол-центра «СберегиЗдоровье».\n\n"
        "Команды:\n"
        "/reset — очистить историю\n"
        "/reload — перечитать контекст\n\n"
        "Пишите вопрос оператора одним сообщением."
    )
    await update.message.reply_text(text)  # type: ignore[union-attr]


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    memory: ChatMemory = context.application.bot_data["memory"]
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    memory.reset(chat_id)
    await update.message.reply_text("История очищена.")  # type: ignore[union-attr]


async def cmd_reload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg = context.application.bot_data["config"]
    context_blob = load_context(cfg.context_dir, cfg.context_char_limit)
    context.application.bot_data["context_blob"] = context_blob

    if context_blob:
        msg = f"Контекст обновлён. Символов: {len(context_blob)}."
    else:
        msg = "Контекст обновлён, но данных нет (папка пуста или не найдена)."

    await update.message.reply_text(msg)  # type: ignore[union-attr]


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    cfg = context.application.bot_data["config"]
    memory: ChatMemory = context.application.bot_data["memory"]
    context_blob: str = context.application.bot_data["context_blob"]

    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    user_text = update.message.text.strip()

    # Context supplement: for EEG questions, inject extracted catalog facts so the LLM can't miss them.
    augmented_context = context_blob
    if "ЭЭГ" in user_text.upper():
        eeg_facts = extract_eeg_facts(cfg.context_dir)
        if eeg_facts:
            augmented_context = (
                augmented_context
                + "\n\n========================\nВЫЖИМКА ДЛЯ ТЕКУЩЕГО ВОПРОСА (использовать как источник истины)\n========================\n"
                + eeg_facts
            )

    memory.add(chat_id, "user", user_text)
    history = memory.get(chat_id)[:-1]  # exclude just-added user msg; it will be appended in ask_llm

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        assistant_text = await ask_llm(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            model=cfg.openai_model,
            system_prompt=SYSTEM_PROMPT,
            context_blob=augmented_context,
            history=history,
            user_text=user_text,
        )
    except Exception as e:
        logger.exception("LLM call failed")
        assistant_text = f"В контексте нет данных: ошибка запроса к модели ({type(e).__name__})."

    memory.add(chat_id, "assistant", assistant_text)
    await update.message.reply_text(assistant_text)


def main() -> None:
    _ensure_ca_bundle()
    cfg = load_config()
    context_blob = load_context(cfg.context_dir, cfg.context_char_limit)

    # Important: getUpdates (long polling) can occupy one connection for a long time.
    # If the connection pool is too small, sendMessage may time out (bot "types" and stops).
    if cfg.telegram_insecure_skip_verify:
        send_request = HTTPXRequest(connection_pool_size=8, pool_timeout=10.0, httpx_kwargs={"verify": False})
        updates_request = HTTPXRequest(connection_pool_size=1, pool_timeout=10.0, httpx_kwargs={"verify": False})
    else:
        send_request = HTTPXRequest(connection_pool_size=8, pool_timeout=10.0)
        updates_request = HTTPXRequest(connection_pool_size=1, pool_timeout=10.0)

    app_builder = Application.builder().token(cfg.telegram_bot_token)
    app_builder = app_builder.request(send_request).get_updates_request(updates_request)
    app = app_builder.build()
    app.bot_data["config"] = cfg
    app.bot_data["memory"] = ChatMemory(limit=cfg.history_limit)
    app.bot_data["context_blob"] = context_blob

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("reload", cmd_reload))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()

