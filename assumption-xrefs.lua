function Link(link)
  local fragment = link.target:match("#([^#]+)$")
  if fragment == nil or not fragment:match("^asm%-") then
    return nil
  end

  for _, class_name in ipairs(link.classes) do
    if class_name == "quarto-xref" then
      return nil
    end
  end

  link.classes:insert("quarto-xref")
  return link
end