$(document).ready(function () {
    let isDragging = false;
    let startX, startY;
    let x = 0, y = 0;
    let scale = 1;
    const minScale = 0.1;
    const maxScale = 3;

    $("#crop-area").on("mousedown", function (e) {
        isDragging = true;
        startX = e.clientX - x;
        startY = e.clientY - y;
    });

    $(document).on("mousemove", function (e) {
        if (isDragging) {
            x = e.clientX - startX;
            y = e.clientY - startY;

            // Obtenir la largeur et la hauteur de l'image
            const imgWidth = $("#uploaded-image").width() * scale;
            const imgHeight = $("#uploaded-image").height() * scale;

            // Limiter le déplacement de l'image
            const maxX = Math.min(0, 512 - imgWidth);
            const maxY = Math.min(0, 512 - imgHeight);

            x = Math.max(x, maxX);
            y = Math.max(y, maxY);
            x = Math.min(x, 0);
            y = Math.min(y, 0);

            $("#uploaded-image").css({
                left: x + 'px',
                top: y + 'px'
            });

            // Mettre à jour les coordonnées du cadrage
            $("#crop-x").val(-x / scale);
            $("#crop-y").val(-y / scale);
        }
    });

    $(document).on("mouseup", function () {
        isDragging = false;
    });

    // Fonctionnalité de zoom avec la molette de la souris
    $("#crop-area").on("wheel", function (e) {
        e.preventDefault();
        const delta = e.originalEvent.deltaY;
        const img = $("#uploaded-image");

        // Calculer le nouveau scale
        if (delta < 0) {
            scale = Math.min(scale + 0.1, maxScale);
        } else {
            scale = Math.max(scale - 0.1, minScale);
        }

        // Appliquer le nouveau scale
        img.css("transform", `scale(${scale})`);

        // Mettre à jour les dimensions de l'image
        const imgWidth = img.width() * scale;
        const imgHeight = img.height() * scale;

        // Limiter le déplacement de l'image
        const maxX = Math.min(0, 512 - imgWidth);
        const maxY = Math.min(0, 512 - imgHeight);

        x = Math.max(x, maxX);
        y = Math.max(y, maxY);
        x = Math.min(x, 0);
        y = Math.min(y, 0);

        img.css({
            left: x + 'px',
            top: y + 'px'
        });

        // Mettre à jour les coordonnées du cadrage
        $("#crop-x").val(-x / scale);
        $("#crop-y").val(-y / scale);
    });
});
