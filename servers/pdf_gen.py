from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import cm
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
import datetime

from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

logo_path = 'logo.png'

def scaled_image(mammogram_path, desired_width_cm):
    img = PILImage.open(mammogram_path)
    aspect_ratio = img.height / img.width
    width = desired_width_cm * cm
    height = width * aspect_ratio
    return Image(mammogram_path, width=width, height=height)

def generate_radiology_report(output_path, mammogram_path, name, jmbg, probability):
    pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='NormalDejaVu', fontName='DejaVu', fontSize=12, leading=14))
    styles.add(ParagraphStyle(name='TitleDejaVu', fontName='DejaVu', fontSize=14, alignment=TA_RIGHT, spaceAfter=12))
    styles.add(ParagraphStyle(name='BigDejaVu', fontName='DejaVu', fontSize=20, alignment=TA_CENTER, spaceAfter=12))

    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    elements = []

    logo = scaled_image(logo_path, 2)
    logo.hAlign = 'RIGHT'

    header = Paragraph("Razvojno-istraživački institut za veštačku inteligenciju Srbije", styles['TitleDejaVu'])


    spacer = Spacer(0.5*cm, 0)  
    table = Table([[header, spacer, logo]], colWidths=[13.5*cm, 0.5*cm, 5*cm])
    table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (2, 0), (2, 0), 'LEFT'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))

    elements.append(table)

    elements.append(Spacer(1, 24))
    
    title = Paragraph("IZVEŠTAJ VEŠTAČKE INTELIGENCIJE", styles['BigDejaVu'])

    elements.append(title)
    elements.append(Spacer(1, 24))

    elements.append(Paragraph(f"<b>Ime i prezime:</b> {name}", styles['NormalDejaVu']))
    elements.append(Paragraph(f"<b>JMBG:</b> {jmbg}", styles['NormalDejaVu']))
    elements.append(Paragraph(f"<b>Verovatnoća postojanja malignih promena (AI analiza):</b> {probability:.2%}", styles['NormalDejaVu']))

    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Opis nalaza:</b>", styles['NormalDejaVu']))
    elements.append(Paragraph(
        "Analiza je izvršena primenom naprednih metoda veštačke inteligencije razvijenih u Institutu za VI. "
        "Rezultat analize je prikazan na priloženoj slici, gde su izdvojene potencijalno kritične "
        f"regije. Procenjena je verovatnoća maligniteta na {probability:.2%}. Predlažemo dalju dijagnostiku u saradnji sa specijalistom radiologije.",
        styles['NormalDejaVu']
    ))

    elements.append(Spacer(1, 12))

    mammogram_image = scaled_image(mammogram_path, 12)
    mammogram_image.hAlign = 'CENTER'
    elements.append(mammogram_image)

    elements.append(Spacer(1, 24))

    today = datetime.date.today().strftime("%d.%m.%Y.")
    elements.append(Paragraph(f"Datum izveštaja: {today}", styles['NormalDejaVu']))

    doc.build(elements)


