let loadElt = document.getElementById("load");

window.onload = function(){
    if (navigator.browserLanguage){
        var language = navigator.browserLanguage;
    }
    else{
        var language = navigator.language;
    }
    
    if (language.indexOf('en') > -1){
        document.location.href = './lang/en.html';
    }
}

window.addEventListener('load',function(){
    this.setTimeout(function(){
        loadElt.classList.add('loaded');
    },700);
});