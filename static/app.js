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

function modelTable(input) {
  var graphs = input;
  chart = document.querySelector('.model_output');
  Plotly.newPlot(chart,graphs,{});
}

function createCheckBox(labels, containerName, element, option) {
  let row = 2;
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

    const span = document.createElement('span');
    span.setAttribute('class', 'checkmark');

    label.appendChild(document.createTextNode(elem));
    label.appendChild(checkbox);
    label.appendChild(span);

    document.querySelector(containerName).appendChild(label);
    $(label).css(
      "grid-row", row
    );
  });
}

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
  $("html, body").animate({ scrollTop: $(document).height()-$(window).height() });
  box = document.querySelectorAll('#label')
  box.forEach(b => {
    console.log(b);
    b.remove();
  })
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
    if(data.hasOwnProperty('dist')){
      getChartDistribution(JSON.parse(data['dist']));
    }
    if(data.hasOwnProperty('textLength')){
      getChartTextLength(JSON.parse(data['textLength']));
    }
    if(data.hasOwnProperty('wordLength')){
      getChartWordLength(JSON.parse(data['wordLength']));
    }
    if(data.hasOwnProperty('bigram')){
      getBigrams(JSON.parse(data['bigram']));
    }
    if(data.hasOwnProperty('labels')){
      createCheckBox(JSON.parse(data['labels']), '.labels_div', 'label','checkbox');
    }
  });
  event.preventDefault();
});

function printLabels(labels) {
  div =  document.querySelector('.trained_labels_text');
  labels.forEach(l => {
    p = document.createElement('p');
    p.innerHTML = l;
    div.appendChild(p);
  })
}

$(document).ready(function() {
  $(".show_csv").addClass('block_click')
  $(".submit_csv").addClass('block_click')
})

function changeColorSubmit() {
  $(".submit_csv").css (
    "background-color", "#284B63"
  );
  $(".submit_csv").removeClass('block_click')
}

$(".submit_csv").on({
  mouseenter: function() {
    $(".submit_csv").css('background-color', '#3C6E71');
  },
  mouseleave: function() {
    $(".submit_csv").css('background-color', "#284B63");
  }
})

function changeColorShowCSV() {
  $(".show_csv").css (
    "background-color", "#284B63"
  );
  $(".show_csv").removeClass('block_click')
}

$(".show_csv").on({
  mouseenter: function() {
    $(".show_csv").css('background-color', '#3C6E71');
  },
  mouseleave: function() {
    $(".show_csv").css('background-color', "#284B63");
  }
})