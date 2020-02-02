$(function() {
    $('#login-file-btn').click(function() {
        $.ajax({
            type: 'GET',
            url: "http://localhost:5000/journal",
            data: 'sunny',
            contentType: false,
            cache: false,
            processData: false,
            success: function(data) {
                window.location.href = 'http://localhost:5000/journal'; 
            },
        });
    });
});