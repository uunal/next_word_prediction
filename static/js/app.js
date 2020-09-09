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
                $('#text_logbert_base_2x2').val(jsondata['logbert-2x2'])
                $('#text_logbert_base_2x1').val(jsondata['logbert-2x1'])
                $('#text_logbert_6x12x768').val(jsondata['logbert-6x12x768'])
                $('#text_logbert_8x8x512').val(jsondata['logbert-8x8x512'])
                $('#text_logberta_6x12x768').val(jsondata['logberta-6x12x768']) 
                $('#text_logbert_small').val(jsondata['logbert-small'])
                                                                            
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
            $('#mask_text_logbert_base_2x2').val(jsondata['logbert-2x2'])
            $('#mask_text_logbert_base_2x1').val(jsondata['logbert-2x1'])
            $('#mask_text_logbert_6x12x768').val(jsondata['logbert-6x12x768'])
            $('#mask_text_logbert_8x8x512').val(jsondata['logbert-8x8x512'])
            $('#mask_text_logberta_6x12x768').val(jsondata['logberta-6x12x768'])            
            $('#mask_text_logbert_small').val(jsondata['logbert-small'])
             
        }).fail(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
        });
    })
})