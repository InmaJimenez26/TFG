# Aplicación de diagnóstico y clasificación de cáncer de piel

## Manual de instalación

### Requisitos previos
Los únicos requisitos previos con los que debe contar la máquina son tener instalado Python y pip en una versión reciente.   
Si no se tienen, en los siguientes enlaces se explica cómo y donde hacerlo:  
- Python: https://www.python.org/downloads/
- Pip: https://pypi.org/project/pip/

### Instalación
En primer lugar hay que descargarse el proyecto, que está subido a un repositorio de GitHub.  
Se utilizará git para clonarlo. Desde una terminal se ejecuta:  
`$ git clone https://github.com/InmaJimenez26/TFG.git`   
Tras descargarlo, se debe acceder a la carpeta que contiene la aplicación.    
`$ cd TFG/app`    
El siguiente paso es crear un entorno de progración python, y activarlo para comenzar a instalar las dependencias necesarias.  
Se haría con los siguientes comandos:  
`$ python3 -m venv venv`   
`$ source venv/bin/activate # for Windows, use venv\Scripts\activate.bat`   
`$ pip install -r requirements.txt`   

## Ejecución
Y por último, ya solo queda ejecutar el fichero que contiene el código que genera la aplicación:  
`$ python3 app_final.py`   
Tras esto, aparecerá una URL local que se debe abrir en un navegador, y la aplicación estará lista para utilizarse.  

