import smtplib
import os
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import schedule
import time
import json

# Set up logging
logging.basicConfig(filename='email_automation.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Load email configurations from a JSON file
def load_email_config(filename):
    try:
        with open(filename) as json_file:
            config = json.load(json_file)
            return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return None

# Create an email message
def create_email(subject, body, to_address, from_address, attachments=[]):
    try:
        msg = MIMEMultipart()
        msg['From'] = from_address
        msg['To'] = to_address
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        for attachment in attachments:
            if os.path.exists(attachment):
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(open(attachment, 'rb').read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition',
                                f'attachment; filename={os.path.basename(attachment)}')
                msg.attach(part)
            else:
                logging.warning(f"Attachment {attachment} does not exist.")

        return msg.as_string()
    except Exception as e:
        logging.error(f"Error creating email: {e}")
        return None

# Send email
def send_email(subject, body, recipients, config, attachments=[]):
    try:
        from_address = config['smtp']['username']
        server = smtplib.SMTP(config['smtp']['server'], config['smtp']['port'])
        server.starttls()
        server.login(from_address, config['smtp']['password'])

        for recipient in recipients:
            email_message = create_email(subject, body, recipient, from_address, attachments)
            if email_message:
                server.sendmail(from_address, recipient, email_message)
                logging.info(f"Email sent to {recipient}")

        server.quit()
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

# Schedule email sending
def schedule_email(subject, body, recipients, config, attachments=[], schedule_time='10:00'):
    def job():
        send_email(subject, body, recipients, config, attachments)
    schedule.every().day.at(schedule_time).do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    email_config = load_email_config('email_config.json')
    if email_config:
        subject = "Daily Report"
        body = "This is an automated email for your daily report."
        recipients = ['recipient1@example.com', 'recipient2@example.com']
        attachments = ['report1.pdf', 'report2.csv']
        schedule_time = '10:00'  # Set desired time for sending emails

        schedule_email(subject, body, recipients, email_config, attachments, schedule_time)
    else:
        logging.error("Email automation process halted due to configuration load failure.")