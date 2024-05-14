/* button.js*/

$(document).ready(function () {
    $('#send').click(function () {
        var text = $('#articleInput').val();

        if (text.trim() === "") {
            alert("기사를 입력하세요!");
            return;
        }

        console.log("Sending text to server:", text);  // 요청 전송 전 로그 추가

        $.ajax({
            url: 'http://localhost:5000/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ text: text }),
            success: function (response) {
                console.log("Received response from server:", response);  // 응답 로그 추가
                $('#answer').text(response.predicted_class);
            },
            error: function (error) {
                console.log("Error:", error);
                alert("오류가 발생했습니다. 다시 시도해주세요.");
            }
        });
    });
});
