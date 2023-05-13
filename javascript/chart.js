
// 设置画布的宽高和边距
const margin = { top: 20, right: 20, bottom: 30, left: 50 };
const width = 960 - margin.left - margin.right;
const height = 500 - margin.top - margin.bottom;

// 在body中添加svg元素
const svg = d3.select("body").append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
  .append("g")
  .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// 定义x和y比例尺
const x = d3.scaleTime().range([0, width]);
const y = d3.scaleLinear().range([height, 0]);

// 定义x和y坐标轴
const xAxis = d3.axisBottom(x);
const yAxis = d3.axisLeft(y);

// 加载CSV数据
d3.csv("C:\\Users\\user\\OneDrive\\Desktop\\Course\\Seminar\\code\\Data\\Normalization\\split_to_XY\\y_test.csv", (error, data) => {
  if (error) throw error;

  // 格式化数据
  data.forEach(d => {
    d.date = new Date(d.date);
    d.price = +d.price;
  });

  // 设置x和y的定义域
  x.domain(d3.extent(data, d => d.date));
  y.domain([0, d3.max(data, d => d.price)]);

  // 添加x轴
  svg.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0," + height + ")")
    .call(xAxis);

  // 添加y轴
  svg.append("g")
    .attr("class", "y axis")
    .call(yAxis)
    .append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", 6)
    .attr("dy", ".71em")
    .style("text-anchor", "end")
    .text("Price ($)");

  // 添加折线
  svg.append("path")
    .datum(data)
    .attr("class", "line")
    .attr("d", d3.line()
      .x(d => x(d.date))
      .y(d => y(d.price))
    );
});
