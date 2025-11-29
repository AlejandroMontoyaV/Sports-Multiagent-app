import wikipedia
import os

# Deportes
deportes = [
"Fútbol","Baloncesto","Béisbol","Tenis","Atletismo","Natación","Boxeo","Ciclismo","Gimnasia artística",
"Voleibol","Rugby","Golf","Artes marciales mixtas","Taekwondo","Judo","Karate","Esgrima","Lucha olímpica",
"Badminton","Tenis de mesa","Hockey sobre hielo","Hockey sobre césped","Waterpolo","Remo","Canotaje",
"Triatlón","Pentatlón moderno","Levantamiento de pesas","Halterofilia","Surf","Skateboarding","Snowboard",
"Esquí alpino","Esquí de fondo","Patinaje artístico","Patinaje de velocidad","Motociclismo","Automovilismo",
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

# Configurar wikipedia
wikipedia.set_lang("es")

path = "data/raw_docs"

# Crear carpeta
os.makedirs(path, exist_ok=True)

# Descargar y guardar artículos
for d in deportes:
    try:
        page = wikipedia.page(d)
        with open(f"{path}{d}.txt", "w", encoding="utf-8") as f:
            f.write(page.content)
        print("Guardado:", d)
    except Exception as e:
        print("No encontrado:", d, "| Error:", e)
