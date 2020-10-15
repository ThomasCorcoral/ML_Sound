var bouton = document.getElementById("But");

// Affichage du bouton quand l'utilisateur descend de 10px
window.onscroll = function() {scrollFunction()};

function scrollFunction() {
if (document.body.scrollTop > 10 || document.documentElement.scrollTop > 10) {
    bouton.style.display = "block";
} else {
    bouton.style.display = "none";
}
}

// Quand l'utilisateur appuye sur le bouton on remonte en haut
function topFunction() {
document.body.scrollTop = 0;
document.documentElement.scrollTop = 0;
}