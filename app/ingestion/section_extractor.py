def extract_page_range(
	pages: list[dict],
	start_page: int,
	end_page: int,
) -> list[dict]:
	"""Return pages whose document page numbers fall within an inclusive range."""
	if start_page < 1 or end_page < start_page:
		raise ValueError("Page range must start at 1 and end at or after the start page.")

	return [
		page
		for page in pages
		if start_page <= page.get("page", 0) <= end_page
	]
