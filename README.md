# Mimesis - Juego de Simulación 3D

## Descripción

Mimesis es un juego de simulación en tiempo real donde los jugadores asumen el papel de una figura histórica y experimentan sus decisiones, relaciones y eventos desde una perspectiva inmersiva.

## Estructura del Proyecto

```
mimesis/
│
├── index.html                    # Archivo HTML principal
│
├── assets/                       # Recursos multimedia
│   ├── images/                   # Imágenes 2D (fotografías, ilustraciones)
│   ├── icons/                    # Iconos y sprites
│   └── sculptures/               # Modelos 3D de estatuas/sculptures
│
├── css/                          # Hojas de estilo
│   ├── styles.css               # Estilos principales globales
│   ├── layout.css               # Estilos específicos del layout (header, footer, main)
│   └── components.css           # Estilos reutilizables de componentes (botones, tarjetas, etc.)
│
├── js/                           # Scripts JavaScript
│   ├── main.js                  # Punto de entrada y bootstrap del juego
│   ├── navigation.js            # Gestión de navegación entre pantallas
│   ├── camera.js                # Lógica de cámara 3D
│   ├── timer.js                 # Controladores de temporización (turnos, eventos)
│   ├── game.js                  # Lógica principal del juego
│   ├── results.js               # Cálculo y gestión de resultados finales
│   └── ranking.js               # Sistema de tablas de clasificación/ranking
│
└── README.md                     # Documentación del proyecto
```

## Requisitos Técnicos

- Navegador moderno con soporte para WebGL
- Al menos 2GB de RAM disponible
- Tarjeta gráfica compatible con WebGL 2.0

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/mimesis.git
   cd Mimesis
   ```

2. Abre `index.html` en un navegador web moderno (Chrome, Firefox, Edge).

## Funcionalidades Principales

### Sistema de Juego
- **Cámara 3D**: Rotación libre alrededor de escenas históricas
- **Temporizador**: Gestión de turnos y eventos temporizados
- **Sistema de Decisiones**: Toma de decisiones con consecuencias históricas

### Interfaz de Usuario
- Navegación entre diferentes vistas del juego
- Tablas de clasificación en tiempo real
- Visualización de resultados después de cada turno

### Recursos Multimedia
- Imágenes de alta resolución para la interfaz
- Iconos personalizados para estados y eventos
- Modelos 3D de esculturas históricas

## Estructura de Archivos JS

| Archivo | Función |
|---------|---------|
| `main.js` | Inicializa todas las dependencias, configura la escena inicial |
| `navigation.js` | Maneja la navegación entre vistas del juego (SPA) |
| `camera.js` | Controla los controles de cámara y proyecciones 3D |
| `timer.js` | Gestiona el temporizador de turnos y eventos |
| `game.js` | Contiene la lógica principal del estado del juego |
| `results.js` | Calcula resultados después de cada decisión |
| `ranking.js` | Mantiene actualizada la tabla de clasificación |

## Estilo CSS

### styles.css
Estilos globales y reset básico.

### layout.css
Estructura general:
- Header
- Footer
- Layout principal (main content area)

### components.css
Componentes reutilizables:
- Botones
- Tarjetas
- Menús de navegación
- Modal windows
- Tabs y pestañas

## Autores

- Nombre del desarrollador principal
- Otros colaboradores (opcional)

## Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.
