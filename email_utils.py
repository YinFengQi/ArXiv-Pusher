import smtplib
import socket
from email.mime.text import MIMEText

import markdown2
from loguru import logger

from config import EMAIL_SERVER_CONFIG


def send_email(subject: str, content: str, receiver_email: str) -> bool:
    """发送邮件通知

    Args:
        subject: 邮件主题
        content: Markdown 格式的邮件内容
        receiver_email: 接收者邮箱，多个邮箱用逗号分隔

    Returns:
        bool: 发送成功返回 True，失败返回 False
    """
    html_content = markdown2.markdown(
        content, extras=["tables", "latex", "fenced-code-blocks"]
    )
    msg = MIMEText(html_content, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = "arxiv-bot@noaddress.com"
    msg["To"] = receiver_email

    server = None
    try:
        logger.info("正在连接SMTP服务器，发送邮件...")
        server = smtplib.SMTP(
            EMAIL_SERVER_CONFIG["smtp_server"],
            EMAIL_SERVER_CONFIG["smtp_port"],
            timeout=10,
        )
        server.starttls()
        server.login(EMAIL_SERVER_CONFIG["sender"], EMAIL_SERVER_CONFIG["password"])

        # 处理多个接收者
        receivers = (
            receiver_email.split(",") if "," in receiver_email else [receiver_email]
        )
        server.sendmail(EMAIL_SERVER_CONFIG["sender"], receivers, msg.as_string())

        logger.success("邮件发送成功")
        return True

    except socket.timeout:
        logger.warning("连接SMTP服务器超时")
        return False
    except smtplib.SMTPException as e:
        logger.error(
            f"SMTP错误: {e.smtp_error.decode() if hasattr(e, 'smtp_error') else str(e)}"
        )
        return False
    except Exception as e:
        logger.error(f"邮件发送失败: {type(e).__name__} - {str(e)}")
        return False
    finally:
        if server:
            try:
                server.quit()
            except Exception:
                pass
