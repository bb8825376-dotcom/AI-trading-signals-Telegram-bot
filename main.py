# ===== IMPORTS =====
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ===== CONFIG =====
TOKEN = 8615825848:AAEZzN4tq9Uf7roKmIXNMFjR-fgmJEk_5do
CHAT_Id

# ===== MODEL =====
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

model = Model()
scaler = MinMaxScaler()

# ===== TRAIN MODEL =====
def train_model():
    df = yf.download("BTC-USD", period="7d", interval="5m")

    df["return"] = df["Close"].pct_change()
    df.dropna(inplace=True)

    X = df[["Close", "Volume"]].values
    y = (df["return"] > 0).astype(int).values

    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    for epoch in range(50):
        pred = model(X_tensor).squeeze()
        loss = loss_fn(pred, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "model.pth")
    print("Model trained")

# ===== LOAD MODEL =====
def load_model():
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

# ===== SIGNAL =====
def get_signal():
    df = yf.download("BTC-USD", period="1d", interval="5m")

    latest = df[["Close", "Volume"]].iloc[-1].values
    latest_scaled = scaler.fit_transform([latest])

    x = torch.tensor(latest_scaled, dtype=torch.float32)

    pred = model(x).item()

    if pred > 0.6:
        return f"🧠 BUY ({pred:.2f})"
    elif pred < 0.4:
        return f"🧠 SELL ({pred:.2f})"
    else:
        return f"🧠 HOLD ({pred:.2f})"

# ===== TELEGRAM =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot Running 🚀")

async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(get_signal())

# ===== AUTO SIGNAL =====
async def autosignal(context):
    result = get_signal()
    await context.bot.send_message(chat_id=CHAT_ID, text=result)

# ===== RUN =====
def main():
    # Run this only first time, then comment it
    # train_model()

    load_model()

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))

    app.job_queue.run_repeating(autosignal, interval=300)

    print("Bot started...")
    app.run_polling()

if __name__ == "__main__":
    main()
