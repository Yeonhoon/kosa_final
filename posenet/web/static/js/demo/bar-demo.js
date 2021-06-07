Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';

// var data = {{xx|safe}};

var chartColors = {
    col1 : 'rgba(54, 162, 235, 0.2)',
    col2 : 'rgba(54, 162, 235, 0.4)',
    col3 : 'rgba(54, 162, 235, 0.6)',
    col4 : 'rgba(54, 162, 235, 0.8)'
}

var ctx = document.getElementById('myAreaChart');
var myChart = new Chart(ctx, {
    type: 'bar',
    responsive: true,
    maintainAspectRatio: false,
    data: {
    labels: Object.values(data['DATES']),
    datasets: [{
        label: 'SQUAT VOLUME',
        data: Object.values(data['VOLUME']),
        backgroundColor: [
        'rgba(255, 99, 132, 0.2)',
        'rgba(54, 162, 235, 0.2)',
        'rgba(255, 206, 86, 0.2)',
        'rgba(75, 192, 192, 0.2)',
        'rgba(153, 102, 255, 0.2)',
        'rgba(255, 159, 64, 0.2)',
        'rgba(255, 99, 132, 0.2)',
        'rgba(54, 162, 235, 0.2)',
        'rgba(255, 206, 86, 0.2)',
        'rgba(75, 192, 192, 0.2)',
        'rgba(153, 102, 255, 0.2)',
        'rgba(255, 159, 64, 0.2)'
        ],
        borderColor: [
        'rgba(54, 162, 235, 1)'
        ],
        borderWidth: 1
    }]
    },
    options: {
        scales: {
            yAxes: [{
                ticks:{
                    beginAtZero: true
                }
            }]
            // xAxes: [{
            //   scaleLabel : {
            //     display: true,
            //     labelString: 'Dates',
            //     fontColor : 'black'
            //   }
            // }],
            // yAxes:[{
            //   scaleLabel: {
            //     display: true,
            //     labelString: "Volume",
            //     fontColor: 'black'
            //   }
            // }],
        
        }
    }
});

var dataset = myChart.data.datasets[0];
console.log(dataset)
console.log(dataset.back)
for(var i=0; i< dataset.data.length; i++){
    if (dataset.data[i] <50) {
        dataset.backgroundColor[i] = chartColors.col1;
    }
    else if (dataset.data[i] < 75) {
        dataset.backgroundColor[i] = chartColors.col2;
    }
    else if (dataset.data[i] < 100) {
        dataset.backgroundColor[i] = chartColors.col3;
    }
    else {
        dataset.backgroundColor[i] = chartColors.col4;
    }
}
myChart.update();