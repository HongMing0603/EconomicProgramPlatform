<script>
		// generate Avatar
		function generateAvatar(text, foregroundColor, backgroundColor) {
			const canvas = document.createElement("canvas");
			const context = canvas.getContext("2d");


			canvas.width = 200;
			canvas.height = 200;

			// Draw background
			context.fillStyle = backgroundColor;
			context.fillRect(0, 0, canvas.width, canvas.height);

			// Draw text
			context.font = "bold 100px Assistant";
			context.fillStyle = foregroundColor;
			context.textAlign = "center";
			context.textBaseline = "middle";
			context.fillText(text, canvas.width / 2, canvas.height / 2);

			return canvas.toDataURL("image/png");
		}
    
		// 獲取username的字串
		const username = document.getElementById("username").getAttribute("data-username");
		document.getElementById("avatar").src = generateAvatar(username[0], "white", "#009578");
	</script>