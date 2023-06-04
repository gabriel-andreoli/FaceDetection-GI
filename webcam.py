import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

fonte = cv2.FONT_HERSHEY_SIMPLEX
cv2.namedWindow('Face detection - Gestao da Informacao', cv2.WINDOW_NORMAL)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Lista de cursos
cursos = ['Administração', 'Ciências Econômicas', 'Ciências Contábeis', 'Gestão da Informação', 'Agronomia',
          'Análise e Desenvolvimento de Sistemas', 'Arquitetura e Urbanismo', 'Artes Visuais',
          'Biomedicina', 'Ciência da Computação', 'Ciências Biológicas',
          'Ciências Sociais', 'Design de Produto',
          'Design Gráfico', 'Direito', 'Educação Física', 'Enfermagem', 'Engenharia Ambiental', 'Engenharia Civil', 'Engenharia de Bioprocessos e Biotecnologia',
          'Engenharia de Produção', 'Engenharia Elétrica', 'Engenharia Florestal', 'Engenharia Industrial Madeireira',
          'Engenharia Mecânica', 'Engenharia Química', 'Estatística', 'Expressão Gráfica', 'Farmácia',
          'Filosofia', 'Física', 'Fisioterapia', 'Geografia',
          'Gestão da Qualidade', 'Gestão Pública', 'História',
          'Informática Biomédica', 'Jornalismo', 'Matemática',
          'Matemática Industrial', 'Medicina', 'Medicina Veterinária', 'Música', 'Negócios Imobiliários',
          'Nutrição', 'Odontologia', 'Pedagogia', 'Psicologia',
          'Publicidade e Propaganda', 'Química', 'Relações Públicas', 'Terapia Ocupacional']

prob_administracao = 0.12
prob_economia = 0.12
prob_contabeis = 0.12
prob_gestao = 0.40

prob_outros = 1 - (prob_administracao + prob_economia + prob_contabeis + prob_gestao)

probabilidade_outros = prob_outros / (len(cursos) - 4)

probabilidades = [prob_administracao, prob_economia, prob_contabeis, prob_gestao] + [probabilidade_outros] * (len(cursos) - 4)

cap = cv2.VideoCapture(0)

curso = None

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        curso = None

    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      if curso is None:
        curso = np.random.choice(cursos, p=probabilidades)

      frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      draw = ImageDraw.Draw(frame_pil)
      font = ImageFont.truetype("arial.ttf", 25) 
      draw.text((x, y-30), curso, font=font, fill=(36, 255, 12, 0))

      frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow('Face detection - Gestao da Informacao', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()