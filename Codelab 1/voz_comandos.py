import requests
import json
def banner_box(title, color="cyan"):
    colors = {
        "reset":"\033[0m","red":"\033[31m","green":"\033[32m","yellow":"\033[33m",
        "blue":"\033[34m","magenta":"\033[35m","cyan":"\033[36m","white":"\033[37m",
        "bright":"\033[1m"
    }
    c = colors.get(color, "")
    b = colors["bright"]
    r = colors["reset"]
    text = f"  {title}  "
    top = "┏" + "━" * len(text) + "┓"
    mid = f"┃{text}┃"
    bot = "┗" + "━" * len(text) + "┛"
    return f"{b}{c}{top}\n{mid}\n{bot}{r}"
# ╔══════════════════════════════════════════════════════════════╗
# ║ Función: get_word_definition                                  ║
# ║                                                              ║
# ║ Esta función recibe una palabra y consulta la API de la RAE  ║
# ║ para obtener su definición oficial. Retorna el significado   ║
# ║ de forma clara y estructurada.                               ║
# ╚══════════════════════════════════════════════════════════════╝
def get_word_definition(word):
    url = f"https://rae-api.com/api/words/{word}"
    response = requests.get(url)
    if response.status_code == 404:
        print("Palabra no encontrada")
        return None
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None
    
    return filtered_data(response.json())

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ Función: filtered_data                                               ║
# ║                                                                      ║
# ║ Esta función procesa los datos obtenidos de la API de la RAE.        ║
# ║ - Extrae el primer grupo de significados de la palabra consultada.   ║
# ║ - Si existen más significados, también los prepara como "otros".     ║
# ║ - Imprime de forma organizada cada definición encontrada.            ║
# ║ - Limita la visualización de significados adicionales a un máximo    ║
# ║   de 5, para mantener la salida legible.                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

def filtered_data(data):
    meanings = (data['data']['meanings'][0]['senses'])
    if len(data['data']['meanings'])>1:
        other_meanings = (data['data']['meanings'][1]['senses'])
    else:
        other_meanings = None
    for meaning in meanings:
        print(
        banner_box(
            f"Significado #{meaning['meaning_number']}", 
            color="red"
        )
    )
        print(f"Significado #{meaning["meaning_number"]}: {meaning['raw']}")
        if other_meanings and not None:
            print(banner_box("Otros significados"))
            counter = 0
            for other_meaning in other_meanings:
                if counter>4:
                    break
                print(f"Significado #{other_meaning["meaning_number"]}: {other_meaning['raw']}")
                counter += 1

