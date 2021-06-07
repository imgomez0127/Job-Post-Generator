(function($){
    $.get("/haha",function(img){$("#testImg").attr("src",`${img}`);})
    
    })(jQuery)

