/* ==========================================================
   sculptures.js
   Catálogo oficial de las 3 esculturas reales de MIMESIS
   (alineado con el dataset del Parcial 2: Belalcázar, Discóbolo, Shakespeare)
   ========================================================== */

window.MIMESIS_SCULPTURES = [
    {
        id: "belalcazar",
        title: "Sebastián de Belalcázar",
        author: "Victorio Macho · 1937 (Cali)",
        emotion: "Determinación",
        subtitle: "Un brazo extendido hacia el horizonte y el torso firme en dirección al oeste.",
        description:
            "Monumento emblemático de Santiago de Cali fundido en bronce. Su silueta se reconoce por el gesto decidido del brazo extendido señalando el horizonte, la pierna adelantada y la postura erguida.",
        image: "assets/sculptures/belalcazar.jpg",
        fallbackSvg: "assets/sculptures/belalcazar.svg",
        tips: [
            "Extiende un brazo completamente hacia un lado señalando el horizonte.",
            "Mantén el torso erguido y el otro brazo flexionado cerca de la cintura.",
            "Separa ligeramente las piernas para marcar estabilidad."
        ]
    },
    {
        id: "discobolo",
        title: "El Discóbolo",
        author: "Mirón de Eléuteras · c. 455 a. C.",
        emotion: "Tensión y enfoque",
        subtitle: "El instante exacto de máxima torsión antes de liberar el lanzamiento.",
        description:
            "Obra cumbre de la escultura griega clásica que captura el movimiento contenido. El cuerpo forma un arco dinámico: el torso se inclina hacia adelante, un brazo se extiende atrás y el otro cruza hacia la rodilla.",
        image: "assets/sculptures/discobolo.jpg",
        fallbackSvg: "assets/sculptures/discobolo.svg",
        tips: [
            "Inclina el torso hacia adelante y flexiona ligeramente las rodillas.",
            "Extiende el brazo trasero hacia atrás y arriba formando una diagonal.",
            "Lleva el otro brazo hacia la rodilla adelantada y gira la cabeza hacia atrás."
        ]
    },
    {
        id: "shakespeare",
        title: "William Shakespeare",
        author: "Tradición escultórica clásica",
        emotion: "Contemplación",
        subtitle: "Postura frontal serena con la mano elevada en gesto teatral de reflexión.",
        description:
            "Representación dramática asociada al célebre soliloquio teatral. Combina una base estable y frontal con el brazo elevado sosteniendo un objeto imaginario a la altura del hombro.",
        image: "assets/sculptures/shakespeare.jpg",
        fallbackSvg: "assets/sculptures/shakespeare.svg",
        tips: [
            "Mantén el cuerpo de frente y la espalda recta.",
            "Eleva un brazo hacia un costado con la palma abierta hacia arriba.",
            "Deja el otro brazo relajado cerca del torso o detrás de la espalda."
        ]
    }
];

/**
 * Asigna la imagen real (.jpg) y, si todavía no se ha colocado en assets/sculptures/,
 * muestra automáticamente la placa SVG local sin romper la interfaz.
 */
window.setSculptureImage = function setSculptureImage(imgEl, sculpture) {
    if (!imgEl || !sculpture) return;
    imgEl.onerror = () => {
        imgEl.onerror = null;
        if (sculpture.fallbackSvg) {
            imgEl.src = sculpture.fallbackSvg;
        }
    };
    imgEl.src = sculpture.image;
    imgEl.alt = sculpture.title;
};
