from __future__ import annotations

import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from bot.config import load_config
from bot.context import load_context
from bot.llm import ask_llm
from bot.memory import ChatMemory
from bot.system_prompt import SYSTEM_PROMPT


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("sberegi-bot")


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

    memory.add(chat_id, "user", user_text)
    history = memory.get(chat_id)[:-1]  # exclude just-added user msg; it will be appended in ask_llm

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        assistant_text = await ask_llm(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            model=cfg.openai_model,
            system_prompt=SYSTEM_PROMPT,
            context_blob=context_blob,
            history=history,
            user_text=user_text,
        )
    except Exception as e:
        logger.exception("LLM call failed")
        assistant_text = f"В контексте нет данных: ошибка запроса к модели ({type(e).__name__})."

    memory.add(chat_id, "assistant", assistant_text)
    await update.message.reply_text(assistant_text)


def main() -> None:
    cfg = load_config()
    context_blob = load_context(cfg.context_dir, cfg.context_char_limit)

    app = Application.builder().token(cfg.telegram_bot_token).build()
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

