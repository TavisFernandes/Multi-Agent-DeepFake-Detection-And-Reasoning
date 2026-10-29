"""Generate the STME NMIMS project certificate PDF for VERITAS.

Layout follows the department template: 4 group copies + 1 individual copy per member.
Run: python make_certificate.py  ->  VERITAS_Certificate.pdf next to this script.
"""
from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

PROJECT = (
    "VERITAS – AI-POWERED MEDIA AUTHENTICITY &amp; FORENSIC INTELLIGENCE: "
    "A Reinforcement-Learning-Orchestrated Multi-Agent Framework for Multimodal "
    "Media Authenticity Verification and Explainable Forensic Reporting"
)
# Add honorifics (e.g. "Mr.", "Ms.") here if wanted; they are printed as "<honorific> <name>".
MEMBERS = [("Mr.", "Harshil Kamde"), ("Mr.", "Siddhant Patil"), ("Mr.", "Tavis Fernandes")]
MENTOR = "Dr. Preeti Agarwal"
DEAN = ("Dr. Preeti Gupta", "Associate Dean", "NMIMS, STME")
DATE_HTML = "24<super>th</super> October, 2026"
PLACE = "Navi Mumbai"
GROUP_COPIES = 4

W, H = letter
LEFT, RIGHT = 108, W - 71
BODY = ParagraphStyle(
    "body", fontName="Times-Roman", fontSize=12, leading=18.7, alignment=TA_JUSTIFY,
    embeddedHyphenation=1, hyphenationLang=lambda w: []
)


def full_name(member):
    hon, name = member
    # Non-breaking space keeps the honorific on the same line as the first name.
    return f"{hon} {name}" if hon else name


def join_names(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def draw_page(c, members):
    cx = W / 2
    c.setFont("Times-Bold", 16)
    for y, text in ((82, "SCHOOL OF TECHNOLOGY"), (110, "MANAGEMENT & ENGINEERING"),
                    (137, "NMIMS, Navi Mumbai")):
        c.drawCentredString(cx, H - y, text)

    c.setFont("Times-Bold", 19)
    title = "CERTIFICATE"
    tw = c.stringWidth(title, "Times-Bold", 19)
    c.drawCentredString(cx, H - 190, title)
    c.setLineWidth(1)
    c.line(cx - tw / 2, H - 193, cx + tw / 2, H - 193)

    names = join_names([full_name(m) for m in members])
    text = (
        f"This is to certify that the project entitled <font name='Times-Bold'>“{PROJECT}”</font>, has been done by "
        f"<font name='Times-Bold'>{names}</font> under my guidance and supervision &amp; has been submitted in partial "
        "fulfilment of degree of Bachelor’s in Technology in Artificial Intelligence and Data "
        "Science of STME, SVKM’s NMIMS (Deemed-to-be University), Navi Mumbai, India."
    )
    p = Paragraph(text, BODY)
    _, ph = p.wrap(RIGHT - LEFT, H)
    body_top = H - 245
    p.drawOn(c, LEFT, body_top - ph)

    # Signature block sits at the template position unless the body runs long.
    sig_y = min(H - 406, body_top - ph - 60)
    c.setLineWidth(0.8)
    c.line(172, sig_y + 13, 245, sig_y + 13)
    c.line(404, sig_y + 13, 477, sig_y + 13)
    c.setFont("Times-Roman", 12)
    pre = "Project mentor ("
    x = 116
    c.drawString(x, sig_y, pre)
    x += c.stringWidth(pre, "Times-Roman", 12)
    c.setFont("Times-Bold", 12)
    c.drawString(x, sig_y, MENTOR)
    x += c.stringWidth(MENTOR, "Times-Bold", 12)
    c.setFont("Times-Roman", 12)
    c.drawString(x, sig_y, ")")
    c.drawString(363, sig_y, "Examiner (Name and Signature)")

    date_y = sig_y - 41
    dp = Paragraph(f"<b>Date:{DATE_HTML}</b>", BODY)
    dp.wrap(300, 40)
    dp.drawOn(c, LEFT, date_y - 4)
    c.setFont("Times-Bold", 12)
    c.drawString(LEFT, date_y - 16, f"Place: {PLACE}")

    dean_x = 449
    dy = date_y - 63
    c.line(dean_x - 43, dy + 14, dean_x + 43, dy + 14)
    for i, line in enumerate(DEAN):
        c.drawCentredString(dean_x, dy - i * 21, line)


def main():
    from reportlab.pdfgen import canvas

    out = Path(__file__).with_name("VERITAS_Certificate.pdf")
    c = canvas.Canvas(str(out), pagesize=letter)
    c.setTitle("VERITAS Project Certificate")
    for _ in range(GROUP_COPIES):
        draw_page(c, MEMBERS)
        c.showPage()
    for m in MEMBERS:
        draw_page(c, [m])
        c.showPage()
    c.save()
    print(out)


if __name__ == "__main__":
    main()
