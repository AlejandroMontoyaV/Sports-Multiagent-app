import wikipedia
from wikipedia.exceptions import PageError, DisambiguationError
import os
import unicodedata


# Deportes
deportes = [
"Fútbol","Baloncesto","Béisbol","Tenis","Atletismo","Natación","Boxeo","Ciclismo","Gimnasia artística",
"Voleibol","Rugby","Golf","Artes marciales mixtas","Taekwondo","Judo","Karate","Esgrima","Lucha olímpica",
"Badminton","Tenis de mesa","Hockey sobre hielo","Hockey sobre césped","Waterpolo","Remo","Canotaje",
"Triatlón","Pentatlón moderno","Levantamiento de pesas","Halterofilia","Surf","Skateboarding","Snowboard",
"Esquí alpino","Esquí de fondo","patinaje artístico sobre hielo","patinaje artístico sobre ruedas","Patinaje de velocidad",
"Motociclismo","Automovilismo","Patinaje de velocidad sobre hielo", "Patinaje de velocidad sobre hielo en pista corta",
"Patinaje de velocidad sobre patines en línea", "Equitación",
"Ajedrez","Billar","Bolos","Tiro con arco","Tiro deportivo","Windsurf","Kitesurf","Escalada deportiva",
"Parkour","Paracaidismo","Buceo","Polo","Criquet","Softbol","Lacrosse","Ultimate frisbee","Handball",
"Balonmano playa","Paddle","Squash","Rugby 7","Rugby league","Fútbol americano","Fútbol sala",
"Polo acuático","Esquí acuático","Bodyboard","Mountain bike","BMX","Enduro motocross","Automodelismo",
"Orientación","Senderismo deportivo","Marcha atlética","Salto de altura","Salto con pértiga",
"Lanzamiento de jabalina","Lanzamiento de disco","Lanzamiento de martillo","Ecuestre","Doma clásica",
"Rodeo","Pesca deportiva","Arco compuesto","Snowkite","Trineo","Luge","Skeleton","Racquetball","Netball",
"Floorball","Kickboxing","Muay thai","Savate","Capoeira","Espeleología deportiva","Paddle surf",
"Regata de vela","Windsurf slalom","Soft-tennis","Cricket Twenty20","Curling"
]

def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace(" ", "_")                          
    text = text.replace("/", "-")                          
    return text

def base_documents(path: str = "data/raw_docs/"):
    """
    Descarga artículos de Wikipedia sobre deportes y los guarda como archivos de texto.
    """
    # Configurar wikipedia
    wikipedia.set_lang("es")
    # Crear carpeta
    os.makedirs(path, exist_ok=True)

    for deporte in deportes:
        filename = slugify(deporte) + ".txt"
        filepath = os.path.join(path, filename)

        try:
            page = wikipedia.page(deporte, auto_suggest=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(page.content)
            print(f"Guardado: {deporte} → {filename}")

        except DisambiguationError as e:
            print(f"(Desambiguación) No descargado: {deporte}. Opciones: {e.options[:5]}")

        except PageError:
            print(f"Página no encontrada: {deporte}")

        except Exception as e:
            print(f"Error desconocido con {deporte}: {e}")
