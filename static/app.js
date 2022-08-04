/**
 * attach classes to HTML elements on document ready
 */
$(document).ready(function() {
  $(".show_csv").addClass('block_click')
  $(".submit_csv").addClass('block_click')
  $('.select_target input[type="radio"]').click(function(){
    let value = $(this).val();
    $('.select_text input[type="radio"]').removeAttr('disabled');
    $('.select_text span').removeClass('disabled');
    $('.select_text input[value="'+value+'"]').prop('checked', false);
    $('.select_text input[value="'+value+'"]').attr('disabled', 'disabled');
    $('.select_text input[value="'+value+'"]').parent().find('span').addClass('disabled');  
  });
})





/**
 * 
 * @param {Plotly JSON Chart} input 
 * 
 * chart to HTML elements
 */
function getChartDistribution(input) {
  var graphs = input;
  chart = document.querySelector('#chartDistribution');
  Plotly.newPlot(chart,graphs,{});
}

function getChartTextLength(input) {
  var graphs = input;
  chart = document.querySelector('#chartTextLength');
  Plotly.newPlot(chart,graphs,{});
}

function getChartWordLength(input) {
  var graphs = input;
  chart = document.querySelector('#chartWordLength');
  Plotly.newPlot(chart,graphs,{});
}

function getBigrams(input) {
  var graphs = input;
  chart = document.querySelector('#chartBigrams');
  Plotly.newPlot(chart,graphs,{});
}





/**
 * 
 * @param {Trained labels} labels 
 * attach the trained labels to HTML
 */
function printLabels(labels) {
  div =  document.querySelector('.trained_labels_text');
  labels.forEach(l => {
    p = document.createElement('span');
    p.innerHTML = l;
    div.appendChild(p);
  })
}





/**
 * 
 * @param {list of labels} labels 
 * @param {container name of the HTML element} containerName 
 * @param {element is the id of the created checkbox} element 
 * @param {input option such a radio or checkbox} option 
 * creates checkbox or radio button in the given html container with the given labels as values
 * and element name as id
 */
function createCheckBox(labels, containerName, element, option) {
  let row = 0;
  newName = document.createElement('div');
  document.querySelector(containerName).appendChild(newName);
  $(newName).css({
    'min-height':'10em',
    'width': '100%',
    'display': 'grid',
    'grid-row': 3,
    'margin-top': '2vw'
  })
  labels.forEach((elem) => {
    row += 1;
    const id = elem;
    const label = document.createElement('label');
    //label.setAttribute('for',id);
    label.setAttribute('class', 'checkbox_label')
    
    const checkbox = document.createElement('input');
    checkbox.type = option;
    checkbox.name = element;
    checkbox.value = elem;
    checkbox.id = element;
    checkbox.classList.add(element)

    const span = document.createElement('span');
    span.setAttribute('class', 'checkmark');

    label.appendChild(document.createTextNode(elem));
    label.appendChild(checkbox);
    label.appendChild(span);

    newName.appendChild(label);
    $(label).css(
      "grid-row", row
    );
  });
}




/**
 * changes button style
 */
function changeColorSubmit() {
  $(".submit_csv").css (
    "background-color", "#284B63"
  );
  $(".submit_csv").removeClass('block_click')
}

function changeColorShowCSV() {
  $(".show_csv").css (
    "background-color", "#284B63"
  );
  $(".show_csv").removeClass('block_click')
}





/**
 * ajax to flask
 */
$('.show_models').click(function() {
  $('.center_model')
  .css(
    "display", "flex"
  );
  $("html, body").animate({ scrollTop: $(document).height()-$(window).height() });
})

$('.show_charts').click(function(event) {
  $('.chart_div').css('display', 'none')
  $('.model_div').css('display', 'none')
  $('.center_select').css(
    "display", "flex"
  );
  $('.labels_div').children('div').remove();
  $('.models_div').children('div').remove();
  $('.options_div').children('div').remove();

  $("html, body").animate({ scrollTop: $(document).height()-$(window).height() });
  target = document.querySelectorAll('#target');
  text = document.querySelectorAll('#text');
  chart = document.querySelectorAll('#chart');
  targetSelected = [];
  textSelected = [];
  chartSelected = [];

  target.forEach(t => {
    if(t.checked) targetSelected.push(t.value);
  });

  text.forEach(t => {
    if(t.checked) textSelected.push(t.value);
  });

  chart.forEach(t => {
    if(t.checked) chartSelected.push(t.value);
  });

  json_data = {"target":targetSelected, "text": textSelected, "chart": chartSelected};

  $.ajax({
    type: 'POST',
    dataType : 'json',
    contentType: 'application/json',
    url: '/showData',
    data: JSON.stringify(json_data),
    error: function(XMLHttpRequest, textStatus, errorThrown) {
      console.log(errorThrown);
    }
  })
  .done(function(data) {
    $('.center_select').hide()
    $('.chart_div').css('display', 'flex')
    $('.model_div').css('display', 'grid')

    if(data.hasOwnProperty('Distribution')){
      getChartDistribution(JSON.parse(data['Distribution']));
    } else {
      elem = document.getElementById('chartDistribution');
      while (elem.firstChild) {
        elem.removeChild(elem.lastChild);
      }
    }

    if(data.hasOwnProperty('Text Length')){
      getChartTextLength(JSON.parse(data['Text Length']));
    }else {
      elem = document.getElementById('chartTextLength');
      while (elem.firstChild) {
        elem.removeChild(elem.lastChild);
      }
    }

    if(data.hasOwnProperty('Word Length')){
      getChartWordLength(JSON.parse(data['Word Length']));
    }else {
      elem = document.getElementById('chartWordLength');
      while (elem.firstChild) {
        elem.removeChild(elem.lastChild);
      }
    }

    if(data.hasOwnProperty('Bi-Grams')){
      getBigrams(JSON.parse(data['Bi-Grams']));
    }else {
      elem = document.getElementById('chartBigrams');
      while (elem.firstChild) {
        elem.removeChild(elem.lastChild);
      }
    }

    if(data.hasOwnProperty('labels')){
      models = ['SVM', 'Naive Bayes', 'Logistic Regression', 'Random Forest'];
      options = ['Tfidf-Vectorizer', 'N-Gram', 'GloVe', 'BERT'];
      average = ['binary', 'macro', 'micro']
      createCheckBox(JSON.parse(data['labels']), '.labels_div', 'label','checkbox');
      createCheckBox(models, '.models_div', 'model','checkbox');
      createCheckBox(options, '.options_div', 'option','checkbox');
      createCheckBox(average, '.average_div', 'average', 'radio');
    }
  });
  event.preventDefault();
});

/**
 * send input text to backend and pass prediction to table
 */
$('.text_input').keyup(function() {
  let text = $('input[name=text_input]').val();
  if (text.length > 0) {
    $('.test_input table').css('display', 'table');
  } else {
    $('.test_input table').css('display', 'none');
  }
  json_data = {"text":text};
  $.ajax({
    type: 'POST',
    dataType : 'json',
    contentType: 'application/json',
    url: '/test_input',
    data: JSON.stringify(json_data),
    error: function(XMLHttpRequest, textStatus, errorThrown) {
      console.log(errorThrown);
    }
  })
  .done(function(data) {
    $('#text_prediction_result').DataTable({
      pagination: 'bootstrap',
      filter: false,
      paging: false,
      info: false,
      ordering: false,
      data: data,
      destroy: true,
      pageLength: 10,
      columns:[
        {data:'Model'},
        {data:'Prediction'}
      ]
    });
    $("html, body").animate({ scrollTop: $(document).height()-$(window).height()},
                            {duration: 600});
    });
  });





/**
 * hover style on model output table
 */
$('.model_output tbody tr').on({
  mouseenter: function() {
    $('.model_output tbody tr').css('opacity', '0.6');
    $(this).css('opacity', '1.0');
  },
  mouseleave: function() {
    $('.model_output tbody tr').css('opacity', '1.0');
  }
});






/**
 * hover on button
 */
$(".submit_csv").on({
  mouseenter: function() {
    $(".submit_csv").css('background-color', '#3C6E71');
  },
  mouseleave: function() {
    $(".submit_csv").css('background-color', "#284B63");
  }
})

$(".show_csv").on({
  mouseenter: function() {
    $(".show_csv").css('background-color', '#3C6E71');
  },
  mouseleave: function() {
    $(".show_csv").css('background-color', "#284B63");
  }
})