var data = []
var token = ""

jQuery(document).ready(function () {
    var slider = $('#max_words')
    slider.on('change mousemove', function (evt) {
        $('#label_max_words').text('Top k words: ' + slider.val())
    })

    var slider_mask = $('#max_words_mask')
    slider_mask.on('change mousemove', function (evt) {
        $('#label_max_words').text('Top k words: ' + slider_mask.val())
    })

    $('#input_text').on('keyup', function (e) {
        if (e.key == ' ') {
            $.ajax({
                url: '/get_end_predictions',
                type: "post",
                contentType: "application/json",
                dataType: "json",
                data: JSON.stringify({
                    "input_text": $('#input_text').val(),
                    "top_k": slider.val(),
                }),
                beforeSend: function () {
                    $('.overlay').show()
                },
                complete: function () {
                    $('.overlay').hide()
                }
            }).done(function (jsondata, textStatus, jqXHR) {
                console.log(jsondata)
                $('#text_logbert_base_2x1').val(jsondata['logbert-2x1'])
                $('#text_logbert_base_2x').val(jsondata['logbert-2x'])
                $('#text_logbert_base').val(jsondata['logbert'])
                $('#text_logbert_medium').val(jsondata['logbert-medium'])
                $('#text_logbert_small').val(jsondata['logbert-small'])
                $('#text_logbert_mini').val(jsondata['logbert-mini'])               
            }).fail(function (jsondata, textStatus, jqXHR) {
                console.log(jsondata)
            });
        }
    })

    $('#btn-process').on('click', function () {
        $.ajax({
            url: '/get_mask_predictions',
            type: "post",
            contentType: "application/json",
            dataType: "json",
            data: JSON.stringify({
                "input_text": $('#mask_input_text').val(),
                "top_k": slider_mask.val(),
            }),
            beforeSend: function () {
                $('.overlay').show()
            },
            complete: function () {
                $('.overlay').hide()
            }
        }).done(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
            $('#mask_text_logbert_base_2x1').val(jsondata['logbert-2x1'])
            $('#mask_text_logbert_base_2x').val(jsondata['logbert-2x'])
            $('#mask_text_logbert_base').val(jsondata['logbert'])
            $('#mask_text_logbert_medium').val(jsondata['logbert-medium'])
            $('#mask_text_logbert_small').val(jsondata['logbert-small'])
            $('#mask_text_logbert_mini').val(jsondata['logbert-mini'])   
        }).fail(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
        });
    })
})