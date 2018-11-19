from monolab.networks.backbone import Backbone
from monolab.networks.deeplab.deeplabv3plus import DCNNType, DeepLabv3Plus

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def notify_mail(address, subject, message):
    email = 'dlcv2k18monolab@gmail.com'
    password = 'hkR-KFa-ymB-gn2'
    send_to_email = address
    subject = subject
    message = message

    msg = MIMEMultipart()
    msg['From'] = email
    msg['To'] = send_to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(message, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(email, password)
    server.sendmail(email, send_to_email, message)
    server.quit()



