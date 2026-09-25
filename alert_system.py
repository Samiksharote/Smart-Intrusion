import asyncio
import telegram
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

class AlertSystem:
    def __init__(self):
        load_dotenv()
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.bot = telegram.Bot(token=self.bot_token)
        self.alert_count = 0
        self.last_alert_time = None
        self.alert_reset_time = timedelta(hours=1)  # Reset alert count after 1 hour
        self.max_alerts = 5  # Maximum number of alerts to send

    async def send_alert(self, message, image_path=None, is_intruder=False):
        """Send alert message and optional image to Telegram"""
        current_time = datetime.now()
        
        # Print debug info
        print(f"Attempting to send alert...")
        print(f"Bot Token: {self.bot_token[:10]}...{self.bot_token[-5:]}")
        print(f"Chat ID: {self.chat_id}")
        
        # Reset alert count if it's been more than reset_time since last alert
        if self.last_alert_time and (current_time - self.last_alert_time) > self.alert_reset_time:
            self.alert_count = 0

        # Check if we've exceeded max alerts for intruder detection
        if is_intruder and self.alert_count >= self.max_alerts:
            print("Maximum alert limit reached. Skipping alert.")
            return False

        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"⚠️ ALERT ⚠️\n{timestamp}\n\n{message}"
        
        try:
            async with self.bot:
                # Add timeout for network operations
                if image_path and os.path.exists(image_path):
                    with open(image_path, 'rb') as photo:
                        await asyncio.wait_for(
                            self.bot.send_photo(
                                chat_id=self.chat_id,
                                photo=photo,
                                caption=full_message
                            ),
                            timeout=30  # 30 second timeout
                        )
                else:
                    await asyncio.wait_for(
                        self.bot.send_message(
                            chat_id=self.chat_id,
                            text=full_message
                        ),
                        timeout=30  # 30 second timeout
                    )

            if is_intruder:
                self.alert_count += 1
                self.last_alert_time = current_time
            
            return True
        except Exception as e:
            print(f"Failed to send alert: {str(e)}")
            return False

    async def send_audio_clip(self, audio_path):
        """Send an audio clip file to the configured Telegram chat as a document."""
        if not audio_path or not os.path.exists(audio_path):
            print("Audio clip not found:", audio_path)
            return False

        try:
            async with self.bot:
                with open(audio_path, 'rb') as doc:
                    await self.bot.send_document(
                        chat_id=self.chat_id,
                        document=doc,
                        caption=f"Audio clip: {os.path.basename(audio_path)}"
                    )
            return True
        except Exception as e:
            print(f"Failed to send audio clip: {e}")
            return False