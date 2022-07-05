function getChartDistribution(input) {
  var graphs = input;
  //graphs.config = {'displayModeBar': false};
  chart = document.querySelector('#chartDistribution');
  Plotly.newPlot(chart,graphs,{});
}

function getChartTextLength(input) {
  var graphs = input;
  //graphs.config = {'displayModeBar': false};
  chart = document.querySelector('#chartTextLength');
  Plotly.newPlot(chart,graphs,{});
}

function getChartWordLength(input) {
  var graphs = input;
  //graphs.config = {'displayModeBar': false};
  chart = document.querySelector('#chartWordLength');
  Plotly.newPlot(chart,graphs,{});
}

function getBigrams(input) {
  var graphs = input;
  //graphs.config = {'displayModeBar': false};
  chart = document.querySelector('#chartBigrams');
  Plotly.newPlot(chart,graphs,{});
}

function createCheckBox(labels) {
  let row = 1;
  labels.forEach((elem) => {
    row += 1;
    const id = elem;
    const label = document.createElement('label');
    //label.setAttribute('for',id);
    label.setAttribute('class', 'checkbox_label')
    
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.name = 'label';
    checkbox.value = elem;
    checkbox.id = 'label';

    const span = document.createElement('span');
    span.setAttribute('class', 'checkmark');

    label.appendChild(document.createTextNode(elem));
    label.appendChild(checkbox);
    label.appendChild(span);

    document.querySelector('.labels_div').appendChild(label);
    $(label).css(
      "grid-row", row
    );
  });
}


function send() {
  labels = document.querySelectorAll("#label");
  l = [];
  labels.forEach(e => {
    if (e.checked) l.push(e.value);
  });
  m = [];
  models = document.querySelectorAll("#model");
  models.forEach(e => {
    if (e.checked) m.push(e.value);
  });
  json_labels = JSON.stringify(l);
  json_models = JSON.stringify(m);
  json_data = {"labels":l, "models": m};
  console.log(json_data);
  $('.center').css(
    "display", "flex"
  );
  $('.model_form').on('submit', function(event) {
    $.ajax({
      type: 'POST',
      dataType : 'json',
      contentType: 'application/json',
      url: '/generateModels',
      data: JSON.stringify(json_data),
      error: function(XMLHttpRequest, textStatus, errorThrown) {
        $('.center').css(
        "display", "none"
        );
        $('#model_output').html('Error: ' + errorThrown); 
      }
    })
    .done(function(data) {
      $('.center').css(
        "display", "none"
      );
      $('#model_output').html(data);
    });
    event.preventDefault();
  });
};

$('.submit_selection').click(function() {
  send();
});

let submitActive = false;
$(document).ready(function() {
  $(".show_csv").addClass('block_click')
  $(".submit_csv").addClass('block_click')
})

function changeColorSubmit() {
  submitActive = true;
  $(".submit_csv").css (
    "background-color", "rgb(0,143,64)"
  );
  $(".submit_csv").removeClass('block_click')
}

$(".submit_csv").on({
  mouseenter: function() {
    if (submitActive) {
      $(".submit_csv").css('background-color', 'rgb(70, 206, 131)');
    } else {
      $(".submit_csv").css('background-color', 'rgb(26, 147, 218)');
    }
  },
  mouseleave: function() {
    if (submitActive) {
      $(".submit_csv").css('background-color', 'rgb(0,143,64)');
    } else {
      $(".submit_csv").css('background-color', ' rgb(0, 90, 143)');
    }
  }
})

function changeColorShowCSV() {
  submitActive = false;
  $(".submit_csv").css (
    "background-color", "rgb(0,90,143)"
  );
  $(".show_csv").css (
    "background-color", "rgb(0,143,64)"
  );
  $(".show_csv").removeClass('block_click')
}

$(".show_csv").on({
  mouseenter: function() {
    $(".show_csv").css('background-color', 'rgb(70, 206, 131)');
  },
  mouseleave: function() {
    $(".show_csv").css('background-color', 'rgb(0,143,64)');
  }
})

