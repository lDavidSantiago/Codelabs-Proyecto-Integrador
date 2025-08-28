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

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Función: get_joke                                                ║
# ║                                                                  ║
# ║ Esta función obtiene un chiste desde la API JokeAPI.             ║
# ║ - Realiza una petición HTTP a la API con idioma español.         ║
# ║ - Si la respuesta es 404, muestra un error indicando que no hay  ║
# ║   chiste disponible.                                             ║
# ║ - Si ocurre otro error, muestra el código de estado recibido.    ║
# ║ - En caso exitoso, retorna el texto del chiste.                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def get_joke():
    url = "https://v2.jokeapi.dev/joke/Any?lang=es&format=txt"
    response = requests.get(url)
    if response.status_code == 404:
        print("Hubo un error al obtener el chiste.")
        return None
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None
    return response.text


def jarvis(app_name):
    import platform
    import subprocess
    import os
    
    sistema = platform.system().lower()
    app_name = app_name.lower().strip()
    
    if sistema == 'windows':
        #Nombre de apps del sistema 
        apps_sistema = {
            'notepad': 'notepad',
            'bloc de notas': 'notepad',
            'calculadora': 'calc',
            'paint': 'mspaint',
            'explorador': 'explorer',
            'explorador de archivos': 'explorer'
        }
        #Rutas comunes de apps instaladas
        apps_instaladas = {
            'chrome': [
                r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'
            ],
            'firefox': [
                r'C:\Program Files\Mozilla Firefox\firefox.exe',
                r'C:\Program Files (x86)\Mozilla Firefox\firefox.exe'
            ],
            'edge': ['msedge'],
            'vscode': [
                f'C:\\Users\\{os.getenv("USERNAME")}\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe',
                r'C:\Program Files\Microsoft VS Code\Code.exe'
            ],
            'visual studio code': [
                f'C:\\Users\\{os.getenv("USERNAME")}\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe',
                r'C:\Program Files\Microsoft VS Code\Code.exe'
            ],
            'word': ['winword'],
            'excel': ['excel'],
            'powerpoint': ['powerpnt'],
            'spotify': [
                f'C:\\Users\\{os.getenv("USERNAME")}\\AppData\\Roaming\\Spotify\\Spotify.exe'
            ],
            'discord': [
                f'C:\\Users\\{os.getenv("USERNAME")}\\AppData\\Local\\Discord\\Update.exe --processStart Discord.exe'
            ],
            'whatsapp': [
                f'C:\\Users\\{os.getenv("USERNAME")}\\AppData\\Local\\WhatsApp\\WhatsApp.exe'
            ],
            'telegram': [
                f'C:\\Users\\{os.getenv("USERNAME")}\\AppData\\Roaming\\Telegram Desktop\\Telegram.exe'
            ]
            ,
            'brave': [
            f'C:\\Users\\{os.getenv("USERNAME")}\\AppData\\Local\\BraveSoftware\\Brave-Browser\\Application\\brave.exe',
            r'C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe',
            r'C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe'
        ]
        }
        
        try:
            if app_name in apps_sistema:
                subprocess.Popen(apps_sistema[app_name], shell=True)
                return f"Abriendo {app_name}"
            
            elif app_name in apps_instaladas:
                rutas = apps_instaladas[app_name]
                
                for ruta in rutas:
                    try:
                        if os.path.exists(ruta.split(' --')[0]):  
                            if '--processStart' in ruta:
                                os.system(f'"{ruta}"')
                            else:
                                subprocess.Popen([ruta], shell=True)
                            return f"Abriendo {app_name}"
                    except:
                        continue
                
                try:
                    subprocess.Popen(rutas[0].split('\\')[-1].split('.exe')[0], shell=True)
                    return f"Abriendo {app_name}"
                except:
                    return f"No encontré {app_name} instalado"
            
            else:
                return f"No conozco la aplicación {app_name}"
                
        except Exception as e:
            return f"Error al abrir {app_name}"
    
    else:
        return "Solo funciona en Windows por ahora"
    
