$("document").ready(function(){
    $("#send").click(function(){
        var message = $("#message").val();

        $.ajax({
            url: "http://localhost:8000/api/",
            type: "POST",
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            //contentType: "application/json",
            //data: JSON.stringify({"message": message})
        }).done(function(data) {
            //console.log(data);
            var testv = JSON.stringify(data);
            if (testv.slice(12, 21) == "no result") {
              document.getElementById("demo").innerHTML = "Your ticket has been successfully received!";
            } else {
              document.getElementById("demo").innerHTML = "Your weekly results: " + testv.slice(12, testv.length - 2);
            }
        });
    });
});

$(function() {
    $('#upload-file-btn').click(function() {
        var form_data = new FormData($('#upload-file')[0]);
        $.ajax({
            type: 'POST',
            url: "http://localhost:8000/uploadajax/",
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function(data) {
                console.log('Success!');
            },
        });
    });
});
