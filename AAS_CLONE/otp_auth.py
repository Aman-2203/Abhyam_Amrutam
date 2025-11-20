import smtplib
import random
import re
import os
from email.message import EmailMessage
from getpass import getpass
from dotenv import load_dotenv



# Email credentials (use environment variables for security)
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

def is_valid_email(email):
    """Check if the email is in a valid format."""
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def generate_otp():
    """Generate a 6-digit random OTP."""
    return str(random.randint(100000, 999999))

def send_otp(receiver_email, otp):
    """Send the OTP to the user's email."""
    msg = EmailMessage()
    msg.set_content(f"Your OTP for login is: {otp}")
    msg["Subject"] = "Your OTP Verification Code"
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        print("✅ OTP sent successfully to your email.")
    except Exception as e:
        print("❌ Failed to send email:", e)
        exit(1)

def main():
    print("=== LOGIN USING EMAIL OTP ===")

    email = input("Enter your email: ").strip()
    if not is_valid_email(email):
        print("❌ Invalid email address.")

        return

    otp = generate_otp()
    send_otp(email, otp)

    user_otp = input("Enter the OTP sent to your email: ").strip()

    if user_otp == otp:
        print("✅ Login successful!")
    else:
        print("❌ Incorrect OTP. Login failed.")

if __name__ == "__main__":
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("Enter your email credentials to send OTPs.")
        SENDER_EMAIL = input("Sender Email: ")
        SENDER_PASSWORD = getpass("Sender Email Password (App password recommended): ")

    main()
